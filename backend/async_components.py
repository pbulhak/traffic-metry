"""Asynchronous wrapper components for TrafficMetry system.

This module provides non-blocking wrapper classes that use ThreadPoolExecutor
to run blocking operations in separate threads, ensuring the main async event
loop remains responsive for WebSocket connections and other I/O operations.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from numpy.typing import NDArray

from backend.camera_stream import CameraStream
from backend.config import CameraSettings, ModelSettings
from backend.detection_models import DetectionResult
from backend.detector import VehicleDetector

logger = logging.getLogger(__name__)


class AsyncCameraStream:
    """Non-blocking wrapper for CameraStream using ThreadPoolExecutor.

    This wrapper runs frame capture operations in a separate thread to prevent
    blocking the main async event loop, which is critical for maintaining
    responsive WebSocket connections.
    """

    def __init__(self, camera_config: CameraSettings, max_workers: int = 2):
        """Initialize async camera stream wrapper.

        Args:
            camera_config: Camera configuration settings
            max_workers: Maximum number of threads for camera operations
        """
        self.camera_config = camera_config
        self.camera_stream: CameraStream | None = None
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="AsyncCamera"
        )
        self.is_connected = False

        # Reconnect mechanism parameters
        self.max_reconnect_attempts = 5
        self.reconnect_delay_base = 1.0  # Seconds
        self.reconnect_delay_max = 60.0
        self.current_reconnect_attempt = 0
        self.last_successful_frame_time = 0.0
        self.connection_timeout_seconds = 30.0

        logger.info(
            f"AsyncCameraStream initialized with {max_workers} worker threads and reconnect capability"
        )

    async def __aenter__(self) -> AsyncCameraStream:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to camera stream in a separate thread."""
        if self.is_connected:
            logger.warning("Camera stream is already connected")
            return

        try:

            def _connect_sync() -> CameraStream:
                camera = CameraStream(self.camera_config)
                camera.__enter__()  # Initialize the connection
                return camera

            loop = asyncio.get_event_loop()
            self.camera_stream = await loop.run_in_executor(self.executor, _connect_sync)
            self.is_connected = True
            logger.info("Async camera stream connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect async camera stream: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect camera stream in a separate thread."""
        if not self.is_connected or not self.camera_stream:
            return

        try:

            def _disconnect_sync(camera: CameraStream) -> None:
                camera.__exit__(None, None, None)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, _disconnect_sync, self.camera_stream)

            self.camera_stream = None
            self.is_connected = False
            logger.info("Async camera stream disconnected")

        except Exception as e:
            logger.error(f"Error disconnecting async camera stream: {e}")
        finally:
            self.executor.shutdown(wait=False)

    async def get_frame(self) -> NDArray | None:
        """Capture frame from camera in a separate thread.

        Returns:
            Video frame as numpy array, or None if capture failed
        """
        if not self.is_connected or not self.camera_stream:
            logger.warning("Camera stream not connected")
            return None

        try:

            def _get_frame_sync(camera: CameraStream) -> NDArray | None:
                return camera.get_frame()

            loop = asyncio.get_event_loop()
            frame = await loop.run_in_executor(self.executor, _get_frame_sync, self.camera_stream)

            if frame is not None:
                import time

                self.last_successful_frame_time = time.time()
                self.current_reconnect_attempt = 0  # Reset on successful frame

            return frame

        except Exception as e:
            logger.error(f"Error capturing frame asynchronously: {e}")
            return None

    async def get_frame_with_reconnect(self) -> NDArray | None:
        """Get frame with automatic reconnection on failure.

        This method implements exponential backoff reconnection strategy
        when frame capture fails due to connection issues.

        Returns:
            Video frame as numpy array, or None if all reconnect attempts failed
        """

        # First attempt - try normal frame capture
        frame = await self.get_frame()
        if frame is not None:
            return frame

        # Check if reconnection should be attempted
        if not self._should_attempt_reconnect():
            return None

        # Attempt reconnection with exponential backoff
        while self.current_reconnect_attempt < self.max_reconnect_attempts:
            self.current_reconnect_attempt += 1

            # Calculate backoff delay with exponential growth
            delay = min(
                self.reconnect_delay_base * (2 ** (self.current_reconnect_attempt - 1)),
                self.reconnect_delay_max,
            )

            logger.warning(
                f"Camera connection lost, attempting reconnect {self.current_reconnect_attempt}/{self.max_reconnect_attempts} "
                f"after {delay:.1f}s delay"
            )

            # Wait before reconnecting
            await asyncio.sleep(delay)

            try:
                # Disconnect and reconnect
                await self.disconnect()
                await self.connect()

                # Test the connection with frame capture
                frame = await self.get_frame()
                if frame is not None:
                    logger.info(
                        f"Camera reconnected successfully after {self.current_reconnect_attempt} attempts"
                    )
                    return frame

            except Exception as e:
                logger.error(f"Reconnection attempt {self.current_reconnect_attempt} failed: {e}")

        # All reconnection attempts failed
        logger.error(f"Failed to reconnect camera after {self.max_reconnect_attempts} attempts")
        return None

    def _should_attempt_reconnect(self) -> bool:
        """Check if reconnection should be attempted based on connection timeout.

        Returns:
            True if reconnection should be attempted, False otherwise
        """
        import time

        if not self.is_connected:
            return True

        # Check if we've exceeded the connection timeout since last successful frame
        if self.last_successful_frame_time > 0:
            time_since_last_frame = time.time() - self.last_successful_frame_time
            if time_since_last_frame > self.connection_timeout_seconds:
                logger.warning(
                    f"No successful frames for {time_since_last_frame:.1f}s "
                    f"(timeout: {self.connection_timeout_seconds}s), initiating reconnect"
                )
                return True

        return False

    def get_camera_info(self) -> dict[str, Any]:
        """Get camera information (non-blocking)."""
        if not self.camera_stream:
            return {"status": "disconnected"}

        import time

        time_since_last_frame = (
            time.time() - self.last_successful_frame_time
            if self.last_successful_frame_time > 0
            else 0
        )

        return {
            "url": str(self.camera_config.url),
            "connected": self.is_connected,
            "worker_threads": self.executor._max_workers,
            "reconnect_status": {
                "current_attempt": self.current_reconnect_attempt,
                "max_attempts": self.max_reconnect_attempts,
                "last_successful_frame_seconds_ago": round(time_since_last_frame, 1),
                "connection_timeout_seconds": self.connection_timeout_seconds,
            },
        }


class AsyncVehicleDetector:
    """Non-blocking wrapper for VehicleDetector using ThreadPoolExecutor.

    This wrapper runs YOLO detection operations in separate threads to prevent
    blocking the main async event loop during heavy model inference.
    """

    def __init__(self, model_config: ModelSettings, max_workers: int = 3):
        """Initialize async vehicle detector wrapper.

        Args:
            model_config: Model configuration settings
            max_workers: Maximum number of threads for detection operations
        """
        self.model_config = model_config
        self.detector: VehicleDetector | None = None
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="AsyncDetector"
        )
        self.is_loaded = False

        logger.info(f"AsyncVehicleDetector initialized with {max_workers} worker threads")

    async def initialize(self) -> None:
        """Initialize vehicle detector in a separate thread."""
        if self.is_loaded:
            logger.warning("Vehicle detector is already loaded")
            return

        try:

            def _initialize_sync() -> VehicleDetector:
                detector = VehicleDetector(self.model_config)
                # Force model loading to ensure it's ready
                _ = detector.get_model_info()
                return detector

            loop = asyncio.get_event_loop()
            self.detector = await loop.run_in_executor(self.executor, _initialize_sync)
            self.is_loaded = True
            logger.info("Async vehicle detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize async vehicle detector: {e}")
            raise

    async def detect_vehicles(self, frame: NDArray) -> list[DetectionResult]:
        """Run vehicle detection on frame in a separate thread.

        Args:
            frame: Video frame to analyze

        Returns:
            List of vehicle detections
        """
        if not self.is_loaded or not self.detector:
            logger.warning("Vehicle detector not initialized")
            return []

        try:

            def _detect_sync(
                detector: VehicleDetector, input_frame: NDArray
            ) -> list[DetectionResult]:
                return detector.detect_vehicles(input_frame)

            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                self.executor, _detect_sync, self.detector, frame
            )

            return detections

        except Exception as e:
            logger.error(f"Error in async vehicle detection: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup detector resources."""
        if self.detector:
            try:

                def _cleanup_sync(detector: VehicleDetector) -> None:
                    # Detector cleanup if needed
                    pass

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, _cleanup_sync, self.detector)

                self.detector = None
                self.is_loaded = False
                logger.info("Async vehicle detector cleaned up")

            except Exception as e:
                logger.error(f"Error cleaning up async detector: {e}")
            finally:
                self.executor.shutdown(wait=False)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information (non-blocking)."""
        if not self.detector:
            return {"status": "not_initialized"}

        try:
            base_info = self.detector.get_model_info()
            base_info["async_worker_threads"] = self.executor._max_workers
            base_info["async_status"] = "loaded" if self.is_loaded else "not_loaded"
            return base_info
        except Exception as e:
            logger.error(f"Error getting async model info: {e}")
            return {"status": "error", "error": str(e)}
