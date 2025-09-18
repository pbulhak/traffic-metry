"""Background processor wrapper for TrafficMetryProcessor integration with FastAPI.

This module provides a wrapper class for running TrafficMetryProcessor
as a background task in FastAPI, enabling real-time event publishing
to WebSocket clients while maintaining the existing detection pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from backend.config import Settings, get_config
from backend.event_publisher import event_publisher
from backend.processor import TrafficMetryProcessor

logger = logging.getLogger(__name__)


class BackgroundProcessor:
    """Background processor wrapper for TrafficMetryProcessor.

    This class runs TrafficMetryProcessor in a background task and
    integrates it with the WebSocket event publishing system.
    """

    def __init__(self, config: Settings | None = None):
        """Initialize background processor.

        Args:
            config: Application configuration. If None, loads from get_config()
        """
        self.config = config or get_config()
        self.processor: TrafficMetryProcessor | None = None
        self.processor_task: asyncio.Task | None = None
        self.is_running = False

    async def start(self) -> None:
        """Start the background processor."""
        if self.is_running:
            logger.warning("Background processor is already running")
            return

        try:
            # Start event publisher first
            await event_publisher.start()
            logger.info("Event publisher started")

            # Create TrafficMetryProcessor with WebSocket integration
            self.processor = TrafficMetryProcessorWithPublishing(self.config)

            # Start processor in background task
            self.processor_task = asyncio.create_task(self.processor.run())
            self.is_running = True

            logger.info("Background processor started successfully")

        except Exception as e:
            logger.error(f"Failed to start background processor: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the background processor."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop processor
        if self.processor:
            self.processor.stop()

        # Wait for processor task to complete
        if self.processor_task:
            try:
                await asyncio.wait_for(self.processor_task, timeout=10.0)
            except TimeoutError:
                logger.warning("Processor task did not stop gracefully, cancelling")
                self.processor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.processor_task

        # Stop event publisher
        await event_publisher.stop()

        logger.info("Background processor stopped")


class TrafficMetryProcessorWithPublishing(TrafficMetryProcessor):
    """Extended TrafficMetryProcessor with WebSocket event publishing.

    This class extends the original TrafficMetryProcessor to publish
    events to WebSocket clients via the EventPublisher using the new
    event-driven architecture.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize processor with publishing capability.

        Args:
            config: Application configuration
        """
        super().__init__(config)
        self.publishing_stats = {"events_published": 0, "publishing_errors": 0}

    async def _process_vehicle_events(self, vehicle_events: list) -> None:
        """Override parent method to add WebSocket event publishing.

        Args:
            vehicle_events: List of VehicleEvent objects (VehicleEntered, VehicleUpdated, VehicleExited)
        """
        # Call parent processing first
        await super()._process_vehicle_events(vehicle_events)

        # Then publish events to WebSocket clients
        for event in vehicle_events:
            try:
                # Convert event to WebSocket format and publish
                websocket_event = event.to_websocket_format()
                if websocket_event:
                    await self._publish_event(websocket_event)

            except Exception as e:
                logger.error(f"Error processing event {type(event).__name__} for publishing: {e}")

    async def _publish_event(self, event: dict[str, Any]) -> None:
        """Publish event to WebSocket clients.

        Args:
            event: Vehicle event in WebSocket format
        """
        try:
            logger.info(f"📤 Publishing event {event.get('eventId')} to WebSocket")  # DEBUG
            success = await event_publisher.publish_event(event)
            if success:
                self.publishing_stats["events_published"] += 1
                logger.info(f"✅ Event {event.get('eventId')} published to WebSocket clients")
            else:
                self.publishing_stats["publishing_errors"] += 1
                logger.warning(f"Failed to queue event {event.get('eventId')} for publishing")

        except Exception as e:
            self.publishing_stats["publishing_errors"] += 1
            logger.error(f"Error publishing event {event.get('eventId')}: {e}")


# Global background processor instance
background_processor = BackgroundProcessor()
