"""Camera stream management for TrafficMetry application.

This module provides camera connection and frame capture functionality using OpenCV
for RTSP streams from IP cameras.
"""

import logging
from typing import Any

import cv2
from numpy.typing import NDArray

from backend.config import CameraSettings

logger = logging.getLogger(__name__)


class CameraConnectionError(Exception):
    """Raised when there's an issue with camera connection or stream access.

    Args:
        message: Error description
        camera_url: Optional camera URL that caused the error
    """

    def __init__(self, message: str, camera_url: str | None = None) -> None:
        super().__init__(message)
        self.camera_url = camera_url


class CameraStream:
    """Manages connection to IP camera RTSP stream and frame capture.

    This class provides a lazy connection pattern where the actual connection
    to the camera is established only when frames are requested.

    Args:
        camera_settings: Configuration settings for camera connection
    """

    def __init__(self, camera_settings: CameraSettings) -> None:
        """Initialize camera stream with given settings.

        Args:
            camera_settings: Configuration containing RTSP URL and video parameters
        """
        self._settings = camera_settings
        self._capture: cv2.VideoCapture | None = None
        self._is_connected = False

        logger.info(f"CameraStream initialized for URL: {camera_settings.url}")

    def connect(self) -> bool:
        """Establish connection to camera RTSP stream.

        Returns:
            True if connection successful, False otherwise

        Raises:
            CameraConnectionError: If connection cannot be established
        """
        try:
            logger.info(f"Connecting to camera: {self._settings.url}")

            # Release any existing connection
            self._cleanup_connection()

            # Create new VideoCapture instance
            self._capture = cv2.VideoCapture(self._settings.url)

            if not self._capture or not self._capture.isOpened():
                raise CameraConnectionError(
                    f"Failed to open camera stream: {self._settings.url}", self._settings.url
                )

            # Configure capture properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)
            self._capture.set(cv2.CAP_PROP_FPS, self._settings.fps)

            # Reduce buffer size for lower latency
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._is_connected = True
            logger.info("Camera connection established successfully")
            return True

        except Exception as e:
            self._cleanup_connection()
            error_msg = f"Camera connection failed: {str(e)}"
            logger.error(error_msg)
            raise CameraConnectionError(error_msg, self._settings.url) from e

    def get_frame(self) -> NDArray | None:
        """Capture a frame from camera stream.

        Uses lazy connection - establishes connection on first call if needed.

        Returns:
            Captured frame as numpy array, or None if capture failed

        Raises:
            CameraConnectionError: If connection cannot be established
        """
        # Lazy connection - connect if not already connected
        if not self._is_connected:
            self.connect()

        if not self._capture or not self._capture.isOpened():
            logger.error("Camera not connected or connection lost")
            return None

        try:
            ret, frame = self._capture.read()

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                return None

            # Validate frame is not empty
            if frame.size == 0:
                logger.warning("Received empty frame from camera")
                return None

            return frame

        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None

    def release(self) -> None:
        """Release camera resources and close connection."""
        logger.info("Releasing camera resources")
        self._cleanup_connection()

    def _cleanup_connection(self) -> None:
        """Internal method to clean up camera connection resources."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as e:
                logger.warning(f"Error releasing camera: {str(e)}")
            finally:
                self._capture = None
                self._is_connected = False

    def __enter__(self) -> "CameraStream":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with automatic cleanup."""
        self.release()
