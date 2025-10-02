"""WebSocket event publisher for real-time TrafficMetry event broadcasting.

This module provides the EventPublisher class for managing WebSocket connections
and broadcasting vehicle detection events to connected frontend clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageCache:
    """LRU cache for serialized WebSocket messages to optimize broadcasting performance."""

    def __init__(self, max_size: int = 500):
        """Initialize message cache.

        Args:
            max_size: Maximum number of cached messages
        """
        self.max_size = max_size
        self.cache: dict[str, str] = {}
        self.access_order: list[str] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def _generate_cache_key(self, event_data: dict[str, Any]) -> str:
        """Generate cache key for event data.

        Args:
            event_data: Event data to generate key for

        Returns:
            Cache key string
        """
        # Create cache key based on event type and critical data
        key_data = {
            "type": event_data.get("type", "unknown"),
            "vehicleId": event_data.get("vehicleId"),
            "vehicleType": event_data.get("vehicleType"),
            "movement_direction": event_data.get("movement", {}).get("direction"),
            "position": event_data.get("position", {}).get("boundingBox"),
        }

        # Create hash from serialized key data
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get_or_create(self, event_data: dict[str, Any]) -> str:
        """Get cached message or create and cache new one.

        Args:
            event_data: Event data to serialize

        Returns:
            Serialized JSON message string
        """
        cache_key = self._generate_cache_key(event_data)

        # Check if message is already cached
        if cache_key in self.cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            logger.debug(f"Cache hit for event key {cache_key}")
            return self.cache[cache_key]

        # Cache miss - create new message
        self._cache_misses += 1
        logger.debug(f"Cache miss for event key {cache_key}")

        try:
            # Directly serialize event data to JSON (no wrapper)
            import json

            serialized_message = json.dumps(event_data)

            # Add to cache
            self.cache[cache_key] = serialized_message
            self.access_order.append(cache_key)

            # Maintain cache size limit (LRU eviction)
            while len(self.cache) > self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
                logger.debug(f"Evicted cache key {oldest_key} (LRU)")

            return serialized_message

        except Exception as e:
            logger.error(f"Failed to create cached message for key {cache_key}: {e}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
        }

    def clear(self) -> None:
        """Clear all cached messages."""
        self.cache.clear()
        self.access_order.clear()
        logger.debug("Message cache cleared")


class ConnectionManager:
    """Manages WebSocket connections for real-time event broadcasting."""

    def __init__(self, max_connections: int = 10, cache_size: int = 500):
        """Initialize connection manager.

        Args:
            max_connections: Maximum number of concurrent WebSocket connections
            cache_size: Maximum number of cached serialized messages
        """
        self.active_connections: set[WebSocket] = set()
        self.max_connections = max_connections
        self.connection_count = 0
        self._connection_lock = asyncio.Lock()
        self.message_cache = MessageCache(cache_size)

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
        """Broadcast vehicle event to all connected clients with optimized caching.

        Args:
            event_data: Vehicle event data in API v2.3 format

        Returns:
            Number of successful broadcasts
        """
        if not self.active_connections:
            logger.debug("No active WebSocket connections for broadcasting")
            return 0

        # Use message cache to get or create serialized JSON
        try:
            cached_json = self.message_cache.get_or_create(event_data)
        except Exception as e:
            logger.error(f"Failed to get/create cached WebSocket message: {e}")
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

    def get_cache_stats(self) -> dict[str, Any]:
        """Get message cache performance statistics.

        Returns:
            Dictionary with cache performance statistics
        """
        return self.message_cache.get_stats()

    def clear_cache(self) -> None:
        """Clear message cache."""
        self.message_cache.clear()


class EventPublisher:
    """High-level event publisher for TrafficMetry WebSocket events."""

    def __init__(self, max_connections: int = 10, cache_size: int = 500):
        """Initialize event publisher.

        Args:
            max_connections: Maximum number of concurrent WebSocket connections
            cache_size: Maximum number of cached serialized messages
        """
        self.connection_manager = ConnectionManager(max_connections, cache_size)
        self.event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=100
        )  # Reduced for better memory management
        self.is_running = False
        self.publisher_task: asyncio.Task | None = None
        self.dropped_events_count = 0  # Track dropped events for monitoring

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
        """Queue event with overflow protection and oldest-event-dropping strategy.

        Args:
            event_data: Vehicle event data in API v2.3 format

        Returns:
            True if event was queued successfully, False if queue was full
        """
        try:
            # Try to add event to queue without blocking
            self.event_queue.put_nowait(event_data)
            return True

        except asyncio.QueueFull:
            # Advanced overflow protection: drop oldest event and add new one
            try:
                # Remove the oldest event from queue
                oldest_event = self.event_queue.get_nowait()
                # Add the new event
                self.event_queue.put_nowait(event_data)

                self.dropped_events_count += 1
                logger.warning(
                    f"Event queue full, dropped oldest event (type: {oldest_event.get('type', 'unknown')}). "
                    f"Total dropped: {self.dropped_events_count}"
                )
                return True

            except (asyncio.QueueEmpty, asyncio.QueueFull):
                # Fallback: just drop the new event if we can't manage the queue
                self.dropped_events_count += 1
                logger.warning(
                    f"Failed to manage event queue overflow. Total dropped: {self.dropped_events_count}"
                )
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

    def get_cache_stats(self) -> dict[str, Any]:
        """Get message cache performance statistics.

        Returns:
            Dictionary with cache performance statistics
        """
        return self.connection_manager.get_cache_stats()

    def get_queue_stats(self) -> dict[str, Any]:
        """Get event queue statistics for monitoring.

        Returns:
            Dictionary with queue performance statistics
        """
        return {
            "queue_size": self.event_queue.qsize(),
            "queue_maxsize": self.event_queue.maxsize,
            "queue_utilization": (self.event_queue.qsize() / self.event_queue.maxsize) * 100,
            "dropped_events_total": self.dropped_events_count,
            "active_connections": len(self.connection_manager.active_connections),
            "max_connections": self.connection_manager.max_connections,
        }

    def clear_cache(self) -> None:
        """Clear message cache."""
        self.connection_manager.clear_cache()

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
