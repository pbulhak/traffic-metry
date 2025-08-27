"""Unit tests for CameraStream class.

These tests use mocking to simulate cv2.VideoCapture behavior without
requiring an actual camera connection.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from backend.camera_stream import CameraConnectionError, CameraStream
from backend.config import CameraSettings


class TestCameraStream:
    """Test suite for CameraStream class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.camera_settings = CameraSettings(
            url="rtsp://test-camera:554/stream", width=1920, height=1080, fps=30
        )

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_init_does_not_connect(self, mock_video_capture: Mock) -> None:
        """Test that __init__ does not immediately connect to camera."""
        camera = CameraStream(self.camera_settings)

        # VideoCapture should not be called during initialization
        mock_video_capture.assert_not_called()
        assert camera._capture is None
        assert not camera._is_connected

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_connect_success(self, mock_video_capture: Mock) -> None:
        """Test successful camera connection."""
        # Setup mock
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)
        result = camera.connect()

        # Verify connection was successful
        assert result is True
        assert camera._is_connected is True
        assert camera._capture is mock_capture_instance

        # Verify OpenCV calls
        mock_video_capture.assert_called_once_with(self.camera_settings.url)
        mock_capture_instance.set.assert_any_call(
            3, self.camera_settings.width
        )  # CAP_PROP_FRAME_WIDTH
        mock_capture_instance.set.assert_any_call(
            4, self.camera_settings.height
        )  # CAP_PROP_FRAME_HEIGHT
        mock_capture_instance.set.assert_any_call(5, self.camera_settings.fps)  # CAP_PROP_FPS
        mock_capture_instance.set.assert_any_call(38, 1)  # CAP_PROP_BUFFERSIZE

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_connect_failure_not_opened(self, mock_video_capture: Mock) -> None:
        """Test connection failure when camera cannot be opened."""
        # Setup mock to fail
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)

        # Should raise CameraConnectionError
        with pytest.raises(CameraConnectionError) as exc_info:
            camera.connect()

        assert "Failed to open camera stream" in str(exc_info.value)
        assert exc_info.value.camera_url == self.camera_settings.url
        assert not camera._is_connected

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_connect_failure_exception(self, mock_video_capture: Mock) -> None:
        """Test connection failure when VideoCapture raises exception."""
        # Setup mock to raise exception
        mock_video_capture.side_effect = Exception("Camera not found")

        camera = CameraStream(self.camera_settings)

        # Should raise CameraConnectionError
        with pytest.raises(CameraConnectionError) as exc_info:
            camera.connect()

        assert "Camera connection failed" in str(exc_info.value)
        assert not camera._is_connected

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_get_frame_lazy_connection(self, mock_video_capture: Mock) -> None:
        """Test that get_frame() triggers lazy connection."""
        # Setup successful mock
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)

        # First call should trigger connection
        frame = camera.get_frame()

        assert frame is not None
        assert camera._is_connected
        mock_video_capture.assert_called_once_with(self.camera_settings.url)

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_get_frame_success(self, mock_video_capture: Mock) -> None:
        """Test successful frame capture."""
        # Setup mock
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)
        camera.connect()

        frame = camera.get_frame()

        assert frame is not None
        assert np.array_equal(frame, test_frame)
        mock_capture_instance.read.assert_called_once()

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_get_frame_read_failure(self, mock_video_capture: Mock) -> None:
        """Test frame capture when read() returns False."""
        # Setup mock to fail read
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (False, None)
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)
        camera.connect()

        frame = camera.get_frame()

        assert frame is None

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_get_frame_empty_frame(self, mock_video_capture: Mock) -> None:
        """Test frame capture when frame is empty."""
        # Setup mock with empty frame
        empty_frame = np.array([])
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, empty_frame)
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)
        camera.connect()

        frame = camera.get_frame()

        assert frame is None

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_get_frame_exception(self, mock_video_capture: Mock) -> None:
        """Test frame capture when read() raises exception."""
        # Setup mock to raise exception
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.side_effect = Exception("Read error")
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)
        camera.connect()

        frame = camera.get_frame()

        assert frame is None

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_release_cleanup(self, mock_video_capture: Mock) -> None:
        """Test proper resource cleanup in release()."""
        # Setup mock
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)
        camera.connect()

        # Verify initial state
        assert camera._is_connected
        assert camera._capture is not None

        # Release resources
        camera.release()

        # Verify cleanup
        mock_capture_instance.release.assert_called_once()
        assert not camera._is_connected
        assert camera._capture is None

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_release_exception_handling(self, mock_video_capture: Mock) -> None:
        """Test release() handles exceptions during cleanup."""
        # Setup mock to raise exception on release
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.release.side_effect = Exception("Release error")
        mock_video_capture.return_value = mock_capture_instance

        camera = CameraStream(self.camera_settings)
        camera.connect()

        # Should not raise exception despite mock error
        camera.release()

        # Cleanup should still occur
        assert not camera._is_connected
        assert camera._capture is None

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_context_manager_success(self, mock_video_capture: Mock) -> None:
        """Test context manager usage with successful operation."""
        # Setup mock
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_capture_instance

        with CameraStream(self.camera_settings) as camera:
            frame = camera.get_frame()
            assert frame is not None
            assert camera._is_connected

        # After context exit, resources should be cleaned up
        mock_capture_instance.release.assert_called_once()

    @patch("backend.camera_stream.cv2.VideoCapture")
    def test_context_manager_exception(self, mock_video_capture: Mock) -> None:
        """Test context manager cleanup when exception occurs."""
        # Setup mock
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture_instance

        with pytest.raises(ValueError), CameraStream(self.camera_settings) as camera:
            camera.connect()
            raise ValueError("Test exception")

        # Resources should still be cleaned up
        mock_capture_instance.release.assert_called_once()

    def test_camera_connection_error_attributes(self) -> None:
        """Test CameraConnectionError stores camera URL."""
        url = "rtsp://test:554/stream"
        error = CameraConnectionError("Test error", url)

        assert str(error) == "Test error"
        assert error.camera_url == url

    def test_camera_connection_error_no_url(self) -> None:
        """Test CameraConnectionError without URL."""
        error = CameraConnectionError("Test error")

        assert str(error) == "Test error"
        assert error.camera_url is None
