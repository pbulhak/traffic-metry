"""OpenCV rendering logic for diagnostics viewer.

This module contains the DiagnosticsRenderer class responsible for all
OpenCV-based visualization and frame rendering operations.
"""

from __future__ import annotations

from typing import Any

import cv2
from numpy.typing import NDArray

from backend.vehicle_events import VehicleEntered, VehicleExited

from .state import SharedControlState
from .utils import DEFAULT_COLORS, DEFAULT_WINDOW_CONFIG


class DiagnosticsRenderer:
    """Pure OpenCV renderer for diagnostics visualization.

    This class handles all frame rendering operations without any thread management.
    It receives data and control state, then produces rendered frames.
    """

    def __init__(self, colors: dict[str, tuple[int, int, int]] | None = None):
        """Initialize renderer with color configuration.

        Args:
            colors: Color configuration dictionary, uses defaults if None
        """
        self.colors = colors or DEFAULT_COLORS
        self.stats_panel_height: int = DEFAULT_WINDOW_CONFIG["stats_panel_height"]  # type: ignore[assignment]

    def create_visualization(
        self,
        frame: NDArray,
        raw_detections: list,
        tracked_detections: list,
        vehicle_events: list,
        stats: dict[str, Any],
        control_state: SharedControlState,
        gui_fps: float = 0.0,
        processing_fps: float = 0.0
    ) -> NDArray:
        """Create complete visualization frame with all overlays.

        Args:
            frame: Input camera frame
            raw_detections: List of raw detection results
            tracked_detections: List of tracked detection results
            vehicle_events: List of vehicle events
            stats: Processing statistics
            control_state: Current control state
            gui_fps: Current GUI FPS
            processing_fps: Current processing FPS

        Returns:
            Rendered frame with all visualizations
        """
        display_frame = frame.copy()

        # Draw raw detections (light gray)
        self._draw_raw_detections(display_frame, raw_detections)

        # Draw tracked detections (green) with conditional labels
        self._draw_tracked_detections(
            display_frame, tracked_detections, control_state
        )

        # Draw vehicle events
        self._draw_vehicle_events(display_frame, vehicle_events)

        # Draw stats panel
        display_frame = self._draw_stats_panel(
            display_frame, stats, control_state, gui_fps, processing_fps
        )

        return display_frame

    def _draw_raw_detections(self, frame: NDArray, raw_detections: list) -> None:
        """Draw raw detection bounding boxes and centroids."""
        for detection in raw_detections:
            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                self.colors["raw_detection"],
                1,
            )
            cv2.circle(
                frame,
                detection.centroid,
                2,
                self.colors["raw_detection"],
                -1
            )

    def _draw_tracked_detections(
        self,
        frame: NDArray,
        tracked_detections: list,
        control_state: SharedControlState
    ) -> None:
        """Draw tracked detection bounding boxes with conditional labels."""
        for detection in tracked_detections:
            # Draw bounding box
            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                self.colors["tracked_vehicle"],
                2,
            )

            # Draw centroid
            cv2.circle(
                frame,
                detection.centroid,
                3,
                self.colors["tracked_vehicle"],
                -1
            )

            # Conditional labels based on control state
            label_y_offset = detection.y1 - 30

            if control_state.show_track_ids and hasattr(detection, "track_id"):
                label = f"Track: {detection.track_id}"
                cv2.putText(
                    frame,
                    label,
                    (detection.x1, label_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.colors["tracked_vehicle"],
                    2,
                )
                label_y_offset += 20

            if control_state.show_confidence:
                conf_label = f"{detection.confidence:.2f}"
                cv2.putText(
                    frame,
                    conf_label,
                    (detection.x1, label_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.colors["tracked_vehicle"],
                    2,
                )

    def _draw_vehicle_events(self, frame: NDArray, vehicle_events: list) -> None:
        """Draw vehicle event indicators."""
        for event in vehicle_events:
            if isinstance(event, VehicleEntered):
                center = event.detection.centroid
                cv2.circle(frame, center, 20, (0, 255, 0), 3)
                cv2.putText(
                    frame,
                    "ENTERED",
                    (center[0] - 30, center[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            elif isinstance(event, VehicleExited):
                if hasattr(event, "journey") and event.journey.best_bbox:
                    x1, y1, x2, y2 = event.journey.best_bbox
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(frame, center, 20, (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        "EXITED",
                        (center[0] - 25, center[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

    def _draw_stats_panel(
        self,
        frame: NDArray,
        stats: dict[str, Any],
        control_state: SharedControlState,
        gui_fps: float,
        processing_fps: float
    ) -> NDArray:
        """Draw enhanced statistics panel overlay with FPS and control state."""
        height, width = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (width, self.stats_panel_height),
            self.colors["panel_bg"],
            -1
        )
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Extract stats
        frame_count = stats.get("frame_count", 0)
        detection_count = stats.get("detection_count", 0)
        tracking_stats = stats.get("tracking_stats", {})

        # Status indicators
        db_status = "ON" if control_state.database_enabled else "OFF"
        cand_status = "ON" if control_state.candidates_enabled else "OFF"
        detect_status = "ON" if control_state.detection_enabled else "OFF"
        track_status = "ON" if control_state.tracking_enabled else "OFF"
        proc_status = "PAUSED" if control_state.processing_paused else "ACTIVE"

        stats_lines = [
            f"FPS: GUI {gui_fps:.1f} | Processing {processing_fps:.1f} | Status: {proc_status}",
            f"Frames: {frame_count} | Detections: {detection_count}",
            f"Active: {tracking_stats.get('active_vehicles', 0)} | Journeys: {tracking_stats.get('total_journeys_completed', 0)}",
            f"Detection: {detect_status} | Tracking: {track_status} | DB: {db_status} | Candidates: {cand_status}",
            "Controls: 1=Detection, 2=Tracking, d=DB, c=Candidates, p=Pause | 'h' for help",
        ]

        # Draw stats text
        for i, line in enumerate(stats_lines):
            cv2.putText(
                frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.colors["stats_text"],
                2,
            )

        return frame

