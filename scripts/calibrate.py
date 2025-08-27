#!/usr/bin/env python3
"""Interactive camera calibration tool for TrafficMetry.

This tool allows users to define lane divider lines and assign traffic directions
by clicking on a live camera feed. The calibration data is saved to config.ini
for use by the main traffic monitoring system.

Usage:
    python scripts/calibrate.py

Interactive Controls:
    - Left click: Add line points (start → end)
    - Right click: Remove last line/cancel current line
    - Enter: Move to direction assignment phase
    - Click lanes: Cycle direction (left → right → stationary)
    - 's': Save configuration
    - 'r': Reset calibration
    - 'q': Quit without saving
"""

import configparser
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.camera_stream import CameraConnectionError, CameraStream
from backend.config import CameraSettings, LaneConfig, get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CalibrationState(Enum):
    """States of the calibration process."""

    DRAWING_LINES = "drawing_lines"
    ASSIGNING_DIRECTIONS = "assigning_directions"
    PREVIEW = "preview"


@dataclass
class Line:
    """Represents a lane divider line."""

    start: tuple[int, int]
    end: tuple[int, int]

    def get_midpoint(self) -> tuple[int, int]:
        """Get the midpoint of the line."""
        return ((self.start[0] + self.end[0]) // 2, (self.start[1] + self.end[1]) // 2)


@dataclass
class Lane:
    """Represents a horizontal traffic lane between two horizontal divider lines."""

    lane_id: int
    left_x: int  # Always 0 (full width)
    right_x: int  # Always frame_width (full width)
    top_y: int  # Y coordinate of top boundary
    bottom_y: int  # Y coordinate of bottom boundary
    direction: str = "stationary"

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is within this horizontal lane."""
        return self.left_x <= x <= self.right_x and self.top_y <= y <= self.bottom_y

    def get_center(self) -> tuple[int, int]:
        """Get the center point of the horizontal lane."""
        return ((self.left_x + self.right_x) // 2, (self.top_y + self.bottom_y) // 2)


class LineManager:
    """Manages lane divider lines and traffic lane definitions."""

    def __init__(self) -> None:
        """Initialize empty line manager."""
        self.lines: list[Line] = []
        self.lane_directions: dict[int, str] = {}

    def add_line(self, line: Line) -> None:
        """Add a new divider line."""
        self.lines.append(line)
        self._recalculate_lanes()
        logger.info(f"Added line from {line.start} to {line.end}")

    def remove_last_line(self) -> bool:
        """Remove the most recently added line."""
        if self.lines:
            removed = self.lines.pop()
            self._recalculate_lanes()
            logger.info(f"Removed line from {removed.start} to {removed.end}")
            return True
        return False

    def clear_all(self) -> None:
        """Clear all lines and lanes."""
        self.lines.clear()
        self.lane_directions.clear()
        logger.info("Cleared all calibration data")

    def _recalculate_lanes(self) -> None:
        """Recalculate lane boundaries after line changes."""
        # Clear existing directions for non-existent lanes
        if len(self.lines) < 2:
            self.lane_directions.clear()
        else:
            # Keep only valid lane IDs
            max_lanes = len(self.lines) - 1
            self.lane_directions = {k: v for k, v in self.lane_directions.items() if k < max_lanes}

    def generate_lanes(self, frame_width: int, frame_height: int) -> dict[int, Lane]:
        """Generate horizontal lane objects between horizontal divider lines."""
        lanes: dict[int, Lane] = {}

        if len(self.lines) < 2:
            return lanes

        # Sort lines by average Y coordinate (top to bottom)
        sorted_lines = sorted(self.lines, key=lambda line: (line.start[1] + line.end[1]) / 2)

        for i in range(len(sorted_lines) - 1):
            top_line = sorted_lines[i]
            bottom_line = sorted_lines[i + 1]

            # Calculate horizontal lane boundaries
            top_y = int((top_line.start[1] + top_line.end[1]) / 2)
            bottom_y = int((bottom_line.start[1] + bottom_line.end[1]) / 2)

            # Lanes span full frame width
            left_x = 0
            right_x = frame_width

            direction = self.lane_directions.get(i, "stationary")
            lanes[i] = Lane(i, left_x, right_x, top_y, bottom_y, direction)

        return lanes

    def get_lane_at_point(self, x: int, y: int, frame_width: int, frame_height: int) -> int | None:
        """Find horizontal lane ID at given coordinates."""
        lanes = self.generate_lanes(frame_width, frame_height)
        for lane_id, lane in lanes.items():
            if lane.contains_point(x, y):
                return lane_id
        return None

    def cycle_lane_direction(self, lane_id: int) -> str:
        """Cycle lane direction: stationary → left → right → stationary."""
        current = self.lane_directions.get(lane_id, "stationary")
        transitions = {"stationary": "left", "left": "right", "right": "stationary"}
        new_direction = transitions[current]
        self.lane_directions[lane_id] = new_direction
        logger.info(f"Lane {lane_id} direction changed: {current} → {new_direction}")
        return new_direction

    def get_config_data(self) -> tuple[list[tuple[int, int, int, int]], dict[int, str]]:
        """Get calibration data in format suitable for config.ini."""
        lines_data = []
        for line in self.lines:
            lines_data.append((line.start[0], line.start[1], line.end[0], line.end[1]))

        return lines_data, self.lane_directions.copy()


class CalibrationUI:
    """Interactive calibration user interface."""

    def __init__(self, camera_settings: CameraSettings):
        """Initialize calibration UI."""
        self.camera_settings = camera_settings
        self.camera_stream = CameraStream(camera_settings)
        self.reference_frame: np.ndarray[Any, np.dtype[Any]] | None = None
        self.line_manager = LineManager()
        self.state = CalibrationState.DRAWING_LINES

        # Mouse interaction state
        self.current_point: tuple[int, int] | None = None
        self.temp_line_end: tuple[int, int] | None = None

        # UI state
        self.message = ""
        self.message_color = (255, 255, 255)  # White default
        self.message_timer = 0

        logger.info("CalibrationUI initialized")

    def run(self) -> None:
        """Main calibration loop."""
        logger.info("Starting camera calibration")

        try:
            if not self._capture_reference_frame():
                logger.error("Failed to capture reference frame")
                return

            self._setup_ui()
            self._main_loop()

        except CameraConnectionError as e:
            logger.error(f"Camera error: {e}")
            self._show_error("Camera connection failed. Check your camera settings.")
        except KeyboardInterrupt:
            logger.info("Calibration interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self._show_error(f"Unexpected error: {e}")
        finally:
            self.camera_stream.release()
            cv2.destroyAllWindows()
            logger.info("Calibration session ended")

    def _capture_reference_frame(self) -> bool:
        """Capture a single reference frame for calibration."""
        logger.info("Capturing reference frame...")

        with self.camera_stream as camera:
            frame = camera.get_frame()
            if frame is None:
                return False

            self.reference_frame = frame.copy()
            logger.info(f"Reference frame captured: {frame.shape}")
            return True

    def _setup_ui(self) -> None:
        """Set up OpenCV window and mouse callback."""
        cv2.namedWindow("TrafficMetry Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TrafficMetry Calibration", 1200, 800)
        cv2.setMouseCallback("TrafficMetry Calibration", self._mouse_callback)

        self._show_message("Calibration started. Click to define lane divider lines.", (0, 255, 0))

    def _main_loop(self) -> None:
        """Main UI event loop."""
        while True:
            # Render current frame
            display_frame = self._render_frame()
            cv2.imshow("TrafficMetry Calibration", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            if not self._handle_keyboard_event(key):
                break

            # Update message timer
            if self.message_timer > 0:
                self.message_timer -= 1
                if self.message_timer == 0:
                    self.message = ""

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse events."""
        if self.state == CalibrationState.DRAWING_LINES:
            self._handle_line_drawing_mouse(event, x, y)
        elif self.state == CalibrationState.ASSIGNING_DIRECTIONS:
            self._handle_direction_assignment_mouse(event, x, y)

    def _handle_line_drawing_mouse(self, event: int, x: int, y: int) -> None:
        """Handle mouse events during line drawing phase."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point is None:
                # First point of new line
                self.current_point = (x, y)
                logger.debug(f"Line start point: ({x}, {y})")
            else:
                # Second point - complete the line
                line = Line(self.current_point, (x, y))
                self.line_manager.add_line(line)
                self.current_point = None
                self.temp_line_end = None

                self._show_message(
                    f"Line added. Total: {len(self.line_manager.lines)} lines", (0, 255, 0)
                )

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Cancel current line or remove last line
            if self.current_point is not None:
                self.current_point = None
                self.temp_line_end = None
                self._show_message("Line cancelled", (255, 255, 0))
            else:
                if self.line_manager.remove_last_line():
                    self._show_message("Last line removed", (255, 255, 0))
                else:
                    self._show_message("No lines to remove", (0, 0, 255))

        elif event == cv2.EVENT_MOUSEMOVE and self.current_point is not None:
            # Update temp line preview
            self.temp_line_end = (x, y)

    def _handle_direction_assignment_mouse(self, event: int, x: int, y: int) -> None:
        """Handle mouse events during direction assignment phase."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.reference_frame is None:
                return
            frame_height, frame_width = self.reference_frame.shape[:2]
            lane_id = self.line_manager.get_lane_at_point(x, y, frame_width, frame_height)

            if lane_id is not None:
                new_direction = self.line_manager.cycle_lane_direction(lane_id)
                self._show_message(
                    f"Horizontal lane {lane_id + 1} direction: {new_direction}", (0, 255, 0)
                )
            else:
                self._show_message(
                    "Click inside a horizontal lane to change direction", (255, 255, 0)
                )

    def _handle_keyboard_event(self, key: int) -> bool:
        """Handle keyboard events. Returns False to exit."""
        if key == ord("q"):
            logger.info("User requested quit")
            return False

        elif key == ord("r"):
            self._reset_calibration()

        elif key == 13:  # Enter
            self._handle_enter_key()

        elif key == ord("s"):
            if self.state == CalibrationState.ASSIGNING_DIRECTIONS:
                if self._save_configuration():
                    self._show_message("Configuration saved successfully!", (0, 255, 0))
                    return False  # Exit after successful save
                else:
                    self._show_error("Failed to save configuration!")
            else:
                self._show_message("Complete line drawing first (press Enter)", (255, 255, 0))

        return True

    def _handle_enter_key(self) -> None:
        """Handle Enter key press."""
        if self.state == CalibrationState.DRAWING_LINES:
            validation_errors = self._validate_lines()
            if not validation_errors:
                self.state = CalibrationState.ASSIGNING_DIRECTIONS
                self.current_point = None
                self.temp_line_end = None
                self._show_message("Now click in lanes to assign directions", (0, 255, 0))
            else:
                self._show_error("; ".join(validation_errors))

    def _render_frame(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Render current frame with overlays."""
        if self.reference_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame = self.reference_frame.copy()

        if self.state == CalibrationState.DRAWING_LINES:
            self._render_line_drawing_mode(frame)
        elif self.state == CalibrationState.ASSIGNING_DIRECTIONS:
            self._render_direction_assignment_mode(frame)

        # Render UI elements
        self._render_instructions(frame)
        self._render_message(frame)

        return frame

    def _render_line_drawing_mode(self, frame: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Render overlays for line drawing mode."""
        # Draw existing lines
        for i, line in enumerate(self.line_manager.lines):
            cv2.line(frame, line.start, line.end, (0, 255, 0), 3)

            # Label lines
            mid_point = line.get_midpoint()
            cv2.putText(
                frame, f"L{i + 1}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        # Draw line being created
        if self.current_point and self.temp_line_end:
            cv2.line(frame, self.current_point, self.temp_line_end, (0, 0, 255), 2)
            cv2.circle(frame, self.current_point, 5, (0, 0, 255), -1)

    def _render_direction_assignment_mode(self, frame: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Render overlays for direction assignment mode with horizontal lanes."""
        frame_height, frame_width = frame.shape[:2]
        lanes = self.line_manager.generate_lanes(frame_width, frame_height)

        # Draw horizontal lane overlays with direction colors
        for lane in lanes.values():
            color = self._get_direction_color(lane.direction)

            # Semi-transparent horizontal lane overlay
            overlay = frame.copy()
            cv2.rectangle(
                overlay, (lane.left_x, lane.top_y), (lane.right_x, lane.bottom_y), color, -1
            )
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

            # Direction label in center of horizontal lane
            center = lane.get_center()
            direction_text = f"{lane.direction.upper()}"
            if lane.direction == "left":
                direction_text += " ←"
            elif lane.direction == "right":
                direction_text += " →"

            cv2.putText(
                frame, direction_text, center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
            )
            cv2.putText(frame, direction_text, center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # Draw horizontal divider lines
        for line in self.line_manager.lines:
            cv2.line(frame, line.start, line.end, (255, 255, 255), 4)
            cv2.line(frame, line.start, line.end, (0, 0, 0), 2)

    def _get_direction_color(self, direction: str) -> tuple[int, int, int]:
        """Get BGR color for traffic direction."""
        colors = {
            "left": (0, 255, 0),  # Green
            "right": (0, 0, 255),  # Red
            "stationary": (0, 255, 255),  # Yellow
        }
        return colors.get(direction, (128, 128, 128))  # Gray default

    def _render_instructions(self, frame: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Render instruction text on frame."""
        if self.state == CalibrationState.DRAWING_LINES:
            instructions = [
                "STEP 1: Define Lane Divider Lines",
                "Left Click: Add line points (start → end)",
                "Right Click: Remove last line / Cancel current",
                "Enter: Move to direction assignment",
                "Q: Quit without saving",
            ]
        elif self.state == CalibrationState.ASSIGNING_DIRECTIONS:
            instructions = [
                "STEP 2: Assign Horizontal Lane Directions",
                "Click in horizontal lane: Cycle direction (stationary→left→right)",
                "Green=LEFT ←, Red=RIGHT →, Yellow=STATIONARY",
                "Traffic moves left/right within horizontal lanes",
                "S: Save configuration and exit",
                "R: Reset calibration | Q: Quit without saving",
            ]
        else:
            instructions = []

        # Render instruction box
        if instructions:
            box_height = len(instructions) * 25 + 20
            cv2.rectangle(frame, (10, 10), (600, box_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (600, box_height), (255, 255, 255), 2)

            for i, instruction in enumerate(instructions):
                y = 35 + i * 25
                cv2.putText(
                    frame, instruction, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )

    def _render_message(self, frame: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Render status message on frame."""
        if self.message:
            # Message background
            text_size = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            frame_height = frame.shape[0]
            msg_y = frame_height - 40

            cv2.rectangle(frame, (10, msg_y - 25), (text_size[0] + 30, msg_y + 10), (0, 0, 0), -1)
            cv2.putText(
                frame,
                self.message,
                (20, msg_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.message_color,
                2,
            )

    def _show_message(self, message: str, color: tuple[int, int, int] = (255, 255, 255)) -> None:
        """Display a temporary status message."""
        self.message = message
        self.message_color = color
        self.message_timer = 60  # ~2 seconds at 30 FPS
        logger.info(f"UI Message: {message}")

    def _show_error(self, message: str) -> None:
        """Display an error message."""
        self._show_message(f"ERROR: {message}", (0, 0, 255))

    def _reset_calibration(self) -> None:
        """Reset all calibration data."""
        self.line_manager.clear_all()
        self.state = CalibrationState.DRAWING_LINES
        self.current_point = None
        self.temp_line_end = None
        self._show_message("Calibration reset", (255, 255, 0))

    def _validate_lines(self) -> list[str]:
        """Validate current line configuration."""
        errors = []

        if len(self.line_manager.lines) < 2:
            errors.append("Need at least 2 divider lines")

        # Check for intersecting lines (basic check)
        lines = self.line_manager.lines
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if self._lines_intersect(lines[i], lines[j]):
                    errors.append(f"Lines {i + 1} and {j + 1} intersect")

        return errors

    def _lines_intersect(self, line1: Line, line2: Line) -> bool:
        """Check if two lines intersect (simplified check)."""
        # Simple bounding box intersection check
        l1_min_x = min(line1.start[0], line1.end[0])
        l1_max_x = max(line1.start[0], line1.end[0])
        l1_min_y = min(line1.start[1], line1.end[1])
        l1_max_y = max(line1.start[1], line1.end[1])

        l2_min_x = min(line2.start[0], line2.end[0])
        l2_max_x = max(line2.start[0], line2.end[0])
        l2_min_y = min(line2.start[1], line2.end[1])
        l2_max_y = max(line2.start[1], line2.end[1])

        # Check if bounding boxes overlap
        return not (
            l1_max_x < l2_min_x or l2_max_x < l1_min_x or l1_max_y < l2_min_y or l2_max_y < l1_min_y
        )

    def _save_configuration(self) -> bool:
        """Save calibration to config.ini file."""
        try:
            lines_data, directions = self.line_manager.get_config_data()

            # Validate we have complete configuration
            if len(lines_data) < 2:
                logger.error("Cannot save: need at least 2 lines")
                return False

            # Create LaneConfig for validation
            LaneConfig(lines=lines_data, directions=directions)

            # Create config.ini file
            config = configparser.ConfigParser()

            # [lanes] section
            config["lanes"] = {}
            for i, (x1, y1, x2, y2) in enumerate(lines_data):
                config["lanes"][f"line_{i + 1}"] = f"{x1},{y1},{x2},{y2}"

            # [directions] section
            config["directions"] = {}
            for lane_id, direction in directions.items():
                config["directions"][str(lane_id)] = direction

            # Write to file
            config_path = Path("config.ini")
            with open(config_path, "w") as f:
                config.write(f)

            logger.info(f"Configuration saved to {config_path}")
            logger.info(f"Lines: {len(lines_data)}, Lanes with directions: {len(directions)}")

            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False


def main() -> None:
    """Main entry point for calibration tool."""
    logger.info("Starting TrafficMetry calibration tool")

    try:
        # Load camera configuration
        config = get_config()
        camera_settings = config.camera

        logger.info(f"Using camera: {camera_settings.url}")
        logger.info(f"Resolution: {camera_settings.width}x{camera_settings.height}")

        # Run calibration UI
        calibration_ui = CalibrationUI(camera_settings)
        calibration_ui.run()

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
