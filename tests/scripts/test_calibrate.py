"""Unit tests for calibration tool.

Tests the LineManager logic, dataclasses, and CalibrationUI functionality
using mocks for OpenCV operations and file I/O.
"""

import configparser

# Import modules under test
import sys
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.config import CameraSettings
from scripts.calibrate import CalibrationState, CalibrationUI, Lane, Line, LineManager


class TestLine:
    """Test Line dataclass."""

    def test_line_creation(self) -> None:
        """Test Line creation with coordinates."""
        line = Line((10, 20), (30, 40))
        assert line.start == (10, 20)
        assert line.end == (30, 40)

    def test_line_midpoint(self) -> None:
        """Test midpoint calculation."""
        line = Line((0, 0), (20, 40))
        assert line.get_midpoint() == (10, 20)

        line2 = Line((5, 10), (15, 30))
        assert line2.get_midpoint() == (10, 20)


class TestLane:
    """Test Lane dataclass."""

    def test_lane_creation(self) -> None:
        """Test Lane creation with boundaries."""
        lane = Lane(0, 0, 1920, 100, 200, "left")
        assert lane.lane_id == 0
        assert lane.left_x == 0
        assert lane.right_x == 1920
        assert lane.top_y == 100
        assert lane.bottom_y == 200
        assert lane.direction == "left"

    def test_lane_contains_point(self) -> None:
        """Test point-in-lane detection."""
        lane = Lane(0, 100, 800, 150, 250)  # Horizontal lane

        # Points inside lane
        assert lane.contains_point(400, 200) is True
        assert lane.contains_point(100, 150) is True  # On boundary
        assert lane.contains_point(800, 250) is True  # On boundary

        # Points outside lane
        assert lane.contains_point(50, 200) is False  # Too far left
        assert lane.contains_point(900, 200) is False  # Too far right
        assert lane.contains_point(400, 100) is False  # Too high
        assert lane.contains_point(400, 300) is False  # Too low

    def test_lane_get_center(self) -> None:
        """Test lane center calculation."""
        lane = Lane(0, 0, 1920, 100, 300)
        assert lane.get_center() == (960, 200)

        lane2 = Lane(1, 200, 800, 50, 150)
        assert lane2.get_center() == (500, 100)


class TestLineManager:
    """Test LineManager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.line_manager = LineManager()

        # Sample horizontal lines (top to bottom)
        self.line1 = Line((100, 150), (800, 160))  # Top line, avg Y = 155
        self.line2 = Line((120, 250), (780, 240))  # Middle line, avg Y = 245
        self.line3 = Line((110, 350), (790, 340))  # Bottom line, avg Y = 345

    def test_empty_initialization(self) -> None:
        """Test LineManager starts empty."""
        assert len(self.line_manager.lines) == 0
        assert len(self.line_manager.lane_directions) == 0

    def test_add_line(self) -> None:
        """Test adding lines to manager."""
        self.line_manager.add_line(self.line1)
        assert len(self.line_manager.lines) == 1
        assert self.line_manager.lines[0] == self.line1

        self.line_manager.add_line(self.line2)
        assert len(self.line_manager.lines) == 2

    def test_remove_last_line(self) -> None:
        """Test removing lines."""
        # Test empty case
        assert self.line_manager.remove_last_line() is False

        # Add and remove
        self.line_manager.add_line(self.line1)
        self.line_manager.add_line(self.line2)

        assert self.line_manager.remove_last_line() is True
        assert len(self.line_manager.lines) == 1
        assert self.line_manager.lines[0] == self.line1

    def test_clear_all(self) -> None:
        """Test clearing all data."""
        self.line_manager.add_line(self.line1)
        self.line_manager.lane_directions[0] = "left"

        self.line_manager.clear_all()
        assert len(self.line_manager.lines) == 0
        assert len(self.line_manager.lane_directions) == 0

    def test_generate_lanes_empty(self) -> None:
        """Test lane generation with no lines."""
        lanes = self.line_manager.generate_lanes(1920, 1080)
        assert len(lanes) == 0

        # One line only
        self.line_manager.add_line(self.line1)
        lanes = self.line_manager.generate_lanes(1920, 1080)
        assert len(lanes) == 0

    def test_generate_lanes_horizontal(self) -> None:
        """Test horizontal lane generation between lines."""
        # Add lines in random order
        self.line_manager.add_line(self.line2)  # Middle (Y=245)
        self.line_manager.add_line(self.line1)  # Top (Y=155)
        self.line_manager.add_line(self.line3)  # Bottom (Y=345)

        lanes = self.line_manager.generate_lanes(1920, 1080)

        # Should create 2 lanes between 3 lines
        assert len(lanes) == 2

        # Lane 0: Between line1 (Y=155) and line2 (Y=245)
        lane0 = lanes[0]
        assert lane0.lane_id == 0
        assert lane0.left_x == 0
        assert lane0.right_x == 1920  # Full width
        assert lane0.top_y == 155  # Top line avg Y
        assert lane0.bottom_y == 245  # Bottom line avg Y
        assert lane0.direction == "stationary"  # Default

        # Lane 1: Between line2 (Y=245) and line3 (Y=345)
        lane1 = lanes[1]
        assert lane1.lane_id == 1
        assert lane1.top_y == 245
        assert lane1.bottom_y == 345

    def test_lane_directions(self) -> None:
        """Test lane direction assignment."""
        self.line_manager.add_line(self.line1)
        self.line_manager.add_line(self.line2)

        # Assign direction
        self.line_manager.lane_directions[0] = "left"

        lanes = self.line_manager.generate_lanes(1920, 1080)
        assert lanes[0].direction == "left"

    def test_get_lane_at_point(self) -> None:
        """Test finding lane by coordinates."""
        self.line_manager.add_line(self.line1)  # Y=155
        self.line_manager.add_line(self.line2)  # Y=245
        self.line_manager.add_line(self.line3)  # Y=345

        # Lane 0: Y 155-245, Lane 1: Y 245-345

        # Points in lane 0
        assert self.line_manager.get_lane_at_point(400, 200, 1920, 1080) == 0
        assert self.line_manager.get_lane_at_point(100, 155, 1920, 1080) == 0  # Top boundary
        assert self.line_manager.get_lane_at_point(800, 245, 1920, 1080) == 0  # Bottom boundary

        # Points in lane 1
        assert self.line_manager.get_lane_at_point(500, 300, 1920, 1080) == 1
        assert self.line_manager.get_lane_at_point(600, 345, 1920, 1080) == 1  # Bottom boundary

        # Points outside all lanes
        assert self.line_manager.get_lane_at_point(400, 100, 1920, 1080) is None  # Above all
        assert self.line_manager.get_lane_at_point(400, 400, 1920, 1080) is None  # Below all

    def test_cycle_lane_direction(self) -> None:
        """Test direction cycling logic."""
        # Test stationary -> left -> right -> stationary cycle
        assert self.line_manager.cycle_lane_direction(0) == "left"
        assert self.line_manager.lane_directions[0] == "left"

        assert self.line_manager.cycle_lane_direction(0) == "right"
        assert self.line_manager.lane_directions[0] == "right"

        assert self.line_manager.cycle_lane_direction(0) == "stationary"
        assert self.line_manager.lane_directions[0] == "stationary"

        # Cycle again
        assert self.line_manager.cycle_lane_direction(0) == "left"

    def test_get_config_data(self) -> None:
        """Test config data export format."""
        self.line_manager.add_line(Line((10, 20), (30, 40)))
        self.line_manager.add_line(Line((50, 60), (70, 80)))
        self.line_manager.lane_directions[0] = "left"
        self.line_manager.lane_directions[1] = "right"

        lines_data, directions = self.line_manager.get_config_data()

        # Verify lines format
        assert lines_data == [(10, 20, 30, 40), (50, 60, 70, 80)]

        # Verify directions format
        assert directions == {0: "left", 1: "right"}


class TestCalibrationUI:
    """Test CalibrationUI functionality with mocks."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.camera_settings = CameraSettings(
            url="rtsp://test:554/stream", width=1920, height=1080, fps=30
        )

    @patch("scripts.calibrate.CameraStream")
    def test_calibration_ui_initialization(self, mock_camera_stream: Mock) -> None:
        """Test CalibrationUI initialization."""
        ui = CalibrationUI(self.camera_settings)

        assert ui.camera_settings == self.camera_settings
        assert ui.reference_frame is None
        assert ui.state == CalibrationState.DRAWING_LINES
        assert ui.current_point is None
        assert ui.message == ""

        # Verify CameraStream was created
        mock_camera_stream.assert_called_once_with(self.camera_settings)

    @patch("scripts.calibrate.cv2")
    @patch("scripts.calibrate.CameraStream")
    def test_capture_reference_frame(self, mock_camera_stream: Mock, mock_cv2: Mock) -> None:
        """Test reference frame capture."""
        # Setup mock camera to return a frame
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_camera_instance = Mock()
        mock_camera_instance.get_frame.return_value = mock_frame
        mock_camera_stream.return_value.__enter__ = Mock(return_value=mock_camera_instance)
        mock_camera_stream.return_value.__exit__ = Mock(return_value=None)

        ui = CalibrationUI(self.camera_settings)
        result = ui._capture_reference_frame()

        assert result is True
        assert ui.reference_frame is not None
        np.testing.assert_array_equal(ui.reference_frame, mock_frame)

    @patch("scripts.calibrate.cv2")
    @patch("scripts.calibrate.CameraStream")
    def test_capture_reference_frame_failure(
        self, mock_camera_stream: Mock, mock_cv2: Mock
    ) -> None:
        """Test reference frame capture failure."""
        # Setup mock camera to return None
        mock_camera_instance = Mock()
        mock_camera_instance.get_frame.return_value = None
        mock_camera_stream.return_value.__enter__ = Mock(return_value=mock_camera_instance)
        mock_camera_stream.return_value.__exit__ = Mock(return_value=None)

        ui = CalibrationUI(self.camera_settings)
        result = ui._capture_reference_frame()

        assert result is False
        assert ui.reference_frame is None

    def test_show_message(self) -> None:
        """Test message display functionality."""
        ui = CalibrationUI(self.camera_settings)

        ui._show_message("Test message", (255, 0, 0))
        assert ui.message == "Test message"
        assert ui.message_color == (255, 0, 0)
        assert ui.message_timer == 60

    def test_show_error(self) -> None:
        """Test error message display."""
        ui = CalibrationUI(self.camera_settings)

        ui._show_error("Test error")
        assert ui.message == "ERROR: Test error"
        assert ui.message_color == (0, 0, 255)  # Red

    def test_reset_calibration(self) -> None:
        """Test calibration reset."""
        ui = CalibrationUI(self.camera_settings)

        # Set up some state
        ui.line_manager.add_line(Line((10, 20), (30, 40)))
        ui.state = CalibrationState.ASSIGNING_DIRECTIONS
        ui.current_point = (100, 200)
        ui.temp_line_end = (300, 400)

        ui._reset_calibration()

        # Verify reset
        assert len(ui.line_manager.lines) == 0
        assert ui.state == CalibrationState.DRAWING_LINES
        assert ui.current_point is None
        assert ui.temp_line_end is None
        assert "reset" in ui.message.lower()

    def test_validate_lines(self) -> None:
        """Test line validation."""
        ui = CalibrationUI(self.camera_settings)

        # Test with no lines
        errors = ui._validate_lines()
        assert "Need at least 2 divider lines" in errors

        # Test with sufficient lines
        ui.line_manager.add_line(Line((0, 100), (1920, 100)))
        ui.line_manager.add_line(Line((0, 200), (1920, 200)))

        errors = ui._validate_lines()
        assert len(errors) == 0  # Should pass validation

    def test_handle_keyboard_event_quit(self) -> None:
        """Test quit keyboard event."""
        ui = CalibrationUI(self.camera_settings)

        # Test 'q' key
        result = ui._handle_keyboard_event(ord("q"))
        assert result is False  # Should exit

    def test_handle_keyboard_event_reset(self) -> None:
        """Test reset keyboard event."""
        ui = CalibrationUI(self.camera_settings)

        # Add some data to reset
        ui.line_manager.add_line(Line((10, 20), (30, 40)))

        # Test 'r' key
        result = ui._handle_keyboard_event(ord("r"))
        assert result is True  # Should continue
        assert len(ui.line_manager.lines) == 0  # Should be reset

    def test_handle_enter_key_insufficient_lines(self) -> None:
        """Test Enter key with insufficient lines."""
        ui = CalibrationUI(self.camera_settings)
        ui.state = CalibrationState.DRAWING_LINES

        # Only one line
        ui.line_manager.add_line(Line((0, 100), (1920, 100)))

        ui._handle_enter_key()

        # Should stay in drawing mode due to validation error
        assert ui.state == CalibrationState.DRAWING_LINES
        assert "Need at least 2" in ui.message

    def test_handle_enter_key_success(self) -> None:
        """Test successful Enter key transition."""
        ui = CalibrationUI(self.camera_settings)
        ui.state = CalibrationState.DRAWING_LINES

        # Add sufficient lines
        ui.line_manager.add_line(Line((0, 100), (1920, 100)))
        ui.line_manager.add_line(Line((0, 200), (1920, 200)))

        ui._handle_enter_key()

        # Should transition to direction assignment
        assert ui.state == CalibrationState.ASSIGNING_DIRECTIONS
        assert ui.current_point is None
        assert ui.temp_line_end is None

    def test_mouse_line_drawing(self) -> None:
        """Test mouse line drawing logic."""
        ui = CalibrationUI(self.camera_settings)
        ui.state = CalibrationState.DRAWING_LINES

        # First click - start point
        ui._handle_line_drawing_mouse(1, 100, 200)  # EVENT_LBUTTONDOWN = 1
        assert ui.current_point == (100, 200)
        assert len(ui.line_manager.lines) == 0  # Line not complete yet

        # Second click - end point
        ui._handle_line_drawing_mouse(1, 300, 400)
        assert ui.current_point is None  # Reset after line creation
        assert len(ui.line_manager.lines) == 1  # Line created

        line = ui.line_manager.lines[0]
        assert line.start == (100, 200)
        assert line.end == (300, 400)

    def test_mouse_right_click_cancel(self) -> None:
        """Test right click to cancel current line."""
        ui = CalibrationUI(self.camera_settings)
        ui.state = CalibrationState.DRAWING_LINES
        ui.current_point = (100, 200)  # Line in progress

        # Right click to cancel
        ui._handle_line_drawing_mouse(2, 300, 400)  # EVENT_RBUTTONDOWN = 2

        assert ui.current_point is None  # Line cancelled
        assert len(ui.line_manager.lines) == 0  # No line created

    def test_get_direction_color(self) -> None:
        """Test direction color mapping."""
        ui = CalibrationUI(self.camera_settings)

        assert ui._get_direction_color("left") == (0, 255, 0)  # Green
        assert ui._get_direction_color("right") == (0, 0, 255)  # Red
        assert ui._get_direction_color("stationary") == (0, 255, 255)  # Yellow
        assert ui._get_direction_color("unknown") == (128, 128, 128)  # Gray default


class TestConfigIntegration:
    """Test config.ini file integration."""

    def test_save_configuration_format(self) -> None:
        """Test config.ini file format generation."""
        line_manager = LineManager()
        line_manager.add_line(Line((100, 150), (800, 160)))
        line_manager.add_line(Line((120, 250), (780, 240)))
        line_manager.lane_directions[0] = "left"

        lines_data, directions = line_manager.get_config_data()

        # Create config manually to test format
        config = configparser.ConfigParser()

        # [lanes] section
        config["lanes"] = {}
        for i, (x1, y1, x2, y2) in enumerate(lines_data):
            config["lanes"][f"line_{i + 1}"] = f"{x1},{y1},{x2},{y2}"

        # [directions] section
        config["directions"] = {}
        for lane_id, direction in directions.items():
            config["directions"][str(lane_id)] = direction

        # Verify format matches expected structure
        assert config["lanes"]["line_1"] == "100,150,800,160"
        assert config["lanes"]["line_2"] == "120,250,780,240"
        assert config["directions"]["0"] == "left"

    @patch("builtins.open", new_callable=mock_open)
    def test_save_configuration_file_operations(self, mock_file: Mock) -> None:
        """Test actual file writing operations."""
        ui = CalibrationUI(CameraSettings())

        # Set up test data
        ui.line_manager.add_line(Line((0, 100), (1920, 100)))
        ui.line_manager.add_line(Line((0, 200), (1920, 200)))
        ui.line_manager.lane_directions[0] = "left"

        # Call save configuration (uses real ConfigParser)
        result = ui._save_configuration()

        assert result is True

        # Verify file was opened for writing (accepts both string and Path)
        mock_file.assert_called_once()
        args, kwargs = mock_file.call_args
        assert str(args[0]) == "config.ini"  # Path or string should resolve to same
        assert args[1] == "w"

        # Verify write was called on the file handle
        mock_file.return_value.write.assert_called()

    @patch("scripts.calibrate.configparser.ConfigParser")
    def test_save_configuration_insufficient_lines(self, mock_config_parser: Mock) -> None:
        """Test save failure with insufficient lines."""
        ui = CalibrationUI(CameraSettings())

        # Only one line - should fail validation
        ui.line_manager.add_line(Line((0, 100), (1920, 100)))

        result = ui._save_configuration()

        assert result is False
        mock_config_parser.assert_not_called()  # Should not attempt to save

    @patch("scripts.calibrate.configparser.ConfigParser")
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_save_configuration_file_error(self, mock_file: Mock, mock_config_parser: Mock) -> None:
        """Test save failure due to file permissions."""
        ui = CalibrationUI(CameraSettings())

        # Set up valid test data
        ui.line_manager.add_line(Line((0, 100), (1920, 100)))
        ui.line_manager.add_line(Line((0, 200), (1920, 200)))

        result = ui._save_configuration()

        assert result is False  # Should handle exception gracefully
