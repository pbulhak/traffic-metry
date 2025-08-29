"""WebSocket event publisher for real-time TrafficMetry event broadcasting.

This module provides the EventPublisher class for managing WebSocket connections
and broadcasting vehicle detection events to connected frontend clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.api_models import WebSocketMessage

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time event broadcasting."""

    def __init__(self, max_connections: int = 10):
        """Initialize connection manager.

        Args:
            max_connections: Maximum number of concurrent WebSocket connections
        """
        self.active_connections: set[WebSocket] = set()
        self.max_connections = max_connections
        self.connection_count = 0
        self._connection_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept

        Returns:
            True if connection accepted, False if rejected (too many connections)
        """
        async with self._connection_lock:
            if len(self.active_connections) >= self.max_connections:
                logger.warning(
                    f"Connection rejected: max connections ({self.max_connections}) reached"
                )
                return False

            try:
                await websocket.accept()
                self.active_connections.add(websocket)
                self.connection_count += 1

                logger.info(
                    f"WebSocket client connected. "
                    f"Active: {len(self.active_connections)}, Total: {self.connection_count}"
                )

                return True

            except Exception as e:
                logger.error(f"Failed to accept WebSocket connection: {e}")
                return False

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Active: {len(self.active_connections)}")

    async def broadcast_event(self, event_data: dict[str, Any]) -> int:
        """Broadcast vehicle event to all connected clients.

        Args:
            event_data: Vehicle event data in API v2.3 format

        Returns:
            Number of successful broadcasts
        """
        if not self.active_connections:
            logger.debug("No active WebSocket connections for broadcasting")
            return 0

        message: dict[str, Any] = {
            "type": "event",
            "data": event_data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Optimize: Serialize JSON once for all clients instead of per-client
        try:
            ws_message = WebSocketMessage(
                type=message["type"],
                data=message["data"],
                timestamp=message["timestamp"]
            )
            cached_json = ws_message.model_dump_json()
        except Exception as e:
            logger.error(f"Failed to serialize WebSocket message: {e}")
            return 0

        successful_sends = 0
        disconnected_connections = []

        for connection in self.active_connections.copy():
            try:
                await self._send_cached_message(connection, cached_json)
                successful_sends += 1

            except WebSocketDisconnect:
                logger.debug("Client disconnected during broadcast")
                disconnected_connections.append(connection)

            except Exception as e:
                logger.warning(f"Failed to send event to WebSocket client: {e}")
                disconnected_connections.append(connection)

        # Clean up disconnected connections
        for connection in disconnected_connections:
            self.disconnect(connection)

        if successful_sends > 0:
            logger.debug(
                f"Event broadcasted to {successful_sends}/{len(self.active_connections) + len(disconnected_connections)} clients"
            )

        return successful_sends

    async def _send_cached_message(self, websocket: WebSocket, cached_json: str) -> None:
        """Send pre-serialized JSON message to a specific WebSocket connection.

        Args:
            websocket: Target WebSocket connection
            cached_json: Pre-serialized JSON message string

        Raises:
            WebSocketDisconnect: If connection is closed
            Exception: If send fails
        """
        await websocket.send_text(cached_json)

    async def _send_to_connection(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        """Send message to a specific WebSocket connection.

        Args:
            websocket: Target WebSocket connection
            message: Message to send

        Raises:
            WebSocketDisconnect: If connection is closed
            Exception: If send fails
        """
        try:
            # Validate message format
            ws_message = WebSocketMessage(**message)
            await websocket.send_text(ws_message.model_dump_json())

        except ValidationError as e:
            logger.error(f"Invalid WebSocket message format: {e}")
            raise

        except Exception as e:
            logger.debug(f"WebSocket send failed: {e}")
            raise


class EventPublisher:
    """High-level event publisher for TrafficMetry WebSocket events."""

    def __init__(self, max_connections: int = 10):
        """Initialize event publisher.

        Args:
            max_connections: Maximum number of concurrent WebSocket connections
        """
        self.connection_manager = ConnectionManager(max_connections)
        self.event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self.is_running = False
        self.publisher_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the event publisher background task."""
        if self.is_running:
            logger.warning("EventPublisher is already running")
            return

        self.is_running = True
        self.publisher_task = asyncio.create_task(self._event_publisher_loop())
        logger.info("EventPublisher started")

    async def stop(self) -> None:
        """Stop the event publisher background task."""
        if not self.is_running:
            return

        self.is_running = False

        if self.publisher_task:
            self.publisher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.publisher_task

        logger.info("EventPublisher stopped")

    async def publish_event(self, event_data: dict[str, Any]) -> bool:
        """Queue a vehicle event for broadcasting.

        Args:
            event_data: Vehicle event data in API v2.3 format

        Returns:
            True if event was queued successfully, False if queue is full
        """
        try:
            self.event_queue.put_nowait(event_data)
            return True

        except asyncio.QueueFull:
            logger.warning("Event queue is full, dropping event")
            return False

    async def connect_client(self, websocket: WebSocket) -> bool:
        """Connect a new WebSocket client.

        Args:
            websocket: WebSocket connection to accept

        Returns:
            True if connection accepted, False if rejected
        """
        return await self.connection_manager.connect(websocket)

    def disconnect_client(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client.

        Args:
            websocket: WebSocket connection to disconnect
        """
        self.connection_manager.disconnect(websocket)

    async def _event_publisher_loop(self) -> None:
        """Background task for processing queued events."""
        logger.info("Event publisher loop started")

        try:
            while self.is_running:
                try:
                    # Wait for event with timeout
                    event_data = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                    # Broadcast to all connected clients
                    successful_sends = await self.connection_manager.broadcast_event(event_data)

                    if successful_sends > 0:
                        logger.debug(f"Event {event_data.get('eventId', 'unknown')} broadcasted")

                    # Mark task as done
                    self.event_queue.task_done()

                except TimeoutError:
                    # No events to process - continue loop
                    continue

                except Exception as e:
                    logger.error(f"Error in event publisher loop: {e}")
                    await asyncio.sleep(1)  # Back off on errors

        except asyncio.CancelledError:
            logger.info("Event publisher loop cancelled")
            raise

        except Exception as e:
            logger.error(f"Event publisher loop failed: {e}")


# Global event publisher instance
event_publisher = EventPublisher()
