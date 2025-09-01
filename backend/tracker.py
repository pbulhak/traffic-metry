"""Vehicle tracking module with event-driven lifecycle management.

This module provides state-of-the-art vehicle tracking using ByteTrack algorithm
from Roboflow Supervision, enhanced with complete vehicle journey lifecycle
management and event-driven architecture.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import supervision as sv

from backend.detection_models import DetectionResult
from backend.vehicle_events import (
    VehicleEntered,
    VehicleEvent,
    VehicleExited,
    VehicleJourney,
    VehicleUpdated,
)

logger = logging.getLogger(__name__)


class VehicleTrackingState:
    """Internal state for a single tracked vehicle journey.

    Manages the complete lifecycle of a vehicle from entry to exit,
    collecting all necessary data for journey analytics.
    """

    def __init__(self, track_id: int, detection: DetectionResult, lane: int | None, direction: str | None):
        """Initialize vehicle tracking state.

        Args:
            track_id: Unique tracking ID assigned by ByteTrack
            detection: Initial detection result
            lane: Initial lane assignment
            direction: Initial movement direction
        """
        self.track_id = track_id
        self.vehicle_type = detection.vehicle_type

        # Journey timestamps
        self.entry_timestamp = detection.frame_timestamp
        self.last_update_timestamp = detection.frame_timestamp

        # Lane tracking
        self.entry_lane = lane
        self.current_lane = lane
        self.lane_changes: list[tuple[float, int, int]] = []  # (timestamp, from_lane, to_lane)

        # Movement tracking
        self.movement_direction = direction

        # Detection history for analytics
        self.confidence_history: list[float] = [detection.confidence]
        self.bbox_history: list[tuple[int, int, int, int]] = [
            (detection.x1, detection.y1, detection.x2, detection.y2)
        ]
        self.detection_timestamps: list[float] = [detection.frame_timestamp]
        self.total_detections = 1

        # Best detection tracking
        self.best_detection = detection
        self.best_confidence = detection.confidence

    def update(self, detection: DetectionResult, lane: int | None, direction: str | None) -> None:
        """Update vehicle state with new detection.

        Args:
            detection: New detection result
            lane: Current lane assignment
            direction: Current movement direction
        """
        # Update detection history
        self.confidence_history.append(detection.confidence)
        self.bbox_history.append((detection.x1, detection.y1, detection.x2, detection.y2))
        self.detection_timestamps.append(detection.frame_timestamp)
        self.total_detections += 1
        self.last_update_timestamp = detection.frame_timestamp

        # Update best detection if confidence improved
        if detection.confidence > self.best_confidence:
            self.best_detection = detection
            self.best_confidence = detection.confidence

        # Track lane changes
        if lane is not None and self.current_lane is not None and lane != self.current_lane:
            self.lane_changes.append((detection.frame_timestamp, self.current_lane, lane))
            logger.debug(f"Vehicle {self.track_id} changed lanes: {self.current_lane} → {lane}")

        self.current_lane = lane
        if direction:  # Update direction if provided
            self.movement_direction = direction

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all detections."""
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def journey_duration_seconds(self) -> float:
        """Calculate journey duration in seconds."""
        return self.last_update_timestamp - self.entry_timestamp

    @property
    def best_bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box of best detection."""
        return (self.best_detection.x1, self.best_detection.y1,
                self.best_detection.x2, self.best_detection.y2)

    def create_journey(self, exit_timestamp: float) -> VehicleJourney:
        """Create complete journey data for database storage.
        
        Args:
            exit_timestamp: When the vehicle exited tracking
            
        Returns:
            Complete VehicleJourney data
        """
        return VehicleJourney(
            track_id=self.track_id,
            vehicle_type=self.vehicle_type,
            entry_timestamp=self.entry_timestamp,
            exit_timestamp=exit_timestamp,
            entry_lane=self.entry_lane,
            exit_lane=self.current_lane,
            lane_changes=self.lane_changes.copy(),
            total_detections=self.total_detections,
            best_confidence=self.best_confidence,
            best_bbox=self.best_bbox,
            best_detection_timestamp=self.best_detection.frame_timestamp,
            movement_direction=self.movement_direction,
            average_confidence=self.average_confidence,
            journey_duration_seconds=exit_timestamp - self.entry_timestamp
        )


class VehicleTrackingManager:
    """Advanced vehicle tracking with complete lifecycle management.
    
    Event-driven architecture that manages vehicle journeys from entry to exit,
    providing real-time updates and complete journey analytics.
    """

    def __init__(self,
                 track_activation_threshold: float = 0.5,
                 lost_track_buffer: int = 30,
                 minimum_matching_threshold: float = 0.8,
                 frame_rate: int = 30,
                 update_interval_seconds: float = 1.0):
        """Initialize vehicle tracking manager.

        Args:
            track_activation_threshold: Detection confidence threshold for track activation
            lost_track_buffer: Number of frames to buffer when a track is lost
            minimum_matching_threshold: Threshold for matching tracks with detections
            frame_rate: Video frame rate for prediction algorithms
            update_interval_seconds: Minimum interval between VehicleUpdated events
        """
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate
            )

            # Vehicle lifecycle management
            self.active_vehicles: dict[int, VehicleTrackingState] = {}
            self.update_interval_seconds = update_interval_seconds
            self.last_update_times: dict[int, float] = {}

            # Statistics
            self.total_vehicles_tracked = 0
            self.total_journeys_completed = 0

            logger.info(
                f"VehicleTrackingManager initialized: "
                f"activation_threshold={track_activation_threshold}, "
                f"lost_buffer={lost_track_buffer}, "
                f"matching_threshold={minimum_matching_threshold}, "
                f"update_interval={update_interval_seconds}s"
            )

        except Exception as e:
            logger.error(f"Failed to initialize VehicleTrackingManager: {e}")
            raise

    def update(self,
               detections: list[DetectionResult],
               lane_assignments: dict[str, tuple[int | None, str | None]]) -> tuple[list[DetectionResult], list[VehicleEvent]]:
        """Update tracking and generate vehicle lifecycle events.

        Args:
            detections: List of raw detections from VehicleDetector
            lane_assignments: Mapping of detection_id to (lane, direction)

        Returns:
            Tuple of (tracked_detections, vehicle_events)
        """
        current_timestamp = time.time()
        events: list[VehicleEvent] = []

        if not detections:
            # Handle empty frame - check for vehicle exits
            events.extend(self._handle_empty_frame(current_timestamp))
            return [], events

        # Convert detections to supervision format
        sv_detections = self._convert_to_supervision_detections(detections)

        # Run ByteTrack tracking
        tracked_sv_detections = self.tracker.update_with_detections(sv_detections)

        # Convert back to DetectionResult with track_id
        tracked_detections = self._convert_from_supervision_detections(
            tracked_sv_detections, detections
        )

        # Generate lifecycle events
        events.extend(self._generate_lifecycle_events(
            tracked_detections, lane_assignments, current_timestamp
        ))

        return tracked_detections, events

    def _generate_lifecycle_events(self,
                                   tracked_detections: list[DetectionResult],
                                   lane_assignments: dict[str, tuple[int | None, str | None]],
                                   current_timestamp: float) -> list[VehicleEvent]:
        """Generate vehicle lifecycle events from tracked detections.
        
        Args:
            tracked_detections: Detections with track_id assigned
            lane_assignments: Lane assignments for each detection
            current_timestamp: Current frame timestamp
            
        Returns:
            List of vehicle events that occurred
        """
        events: list[VehicleEvent] = []
        current_track_ids = set()

        # Process active vehicles
        for detection in tracked_detections:
            if detection.track_id is None:
                continue

            track_id = detection.track_id
            current_track_ids.add(track_id)

            # Get lane assignment
            lane, direction = lane_assignments.get(detection.detection_id, (None, None))

            if track_id not in self.active_vehicles:
                # New vehicle entered
                vehicle_state = VehicleTrackingState(track_id, detection, lane, direction)
                self.active_vehicles[track_id] = vehicle_state
                self.total_vehicles_tracked += 1

                entered_event = VehicleEntered(
                    track_id=track_id,
                    timestamp=detection.frame_timestamp,
                    vehicle_type=detection.vehicle_type,
                    detection=detection,
                    lane=lane,
                    direction=direction
                )
                events.append(entered_event)

                # Set initial update time
                self.last_update_times[track_id] = current_timestamp

                logger.debug(f"Vehicle {track_id} entered: {detection.vehicle_type} in lane {lane}")

            else:
                # Update existing vehicle
                vehicle_state = self.active_vehicles[track_id]
                vehicle_state.update(detection, lane, direction)

                # Generate VehicleUpdated event if enough time has passed
                last_update = self.last_update_times.get(track_id, 0)
                if current_timestamp - last_update >= self.update_interval_seconds:

                    updated_event = VehicleUpdated(
                        track_id=track_id,
                        timestamp=detection.frame_timestamp,
                        vehicle_type=detection.vehicle_type,
                        detection=detection,
                        lane=lane,
                        direction=direction,
                        total_detections_so_far=vehicle_state.total_detections,
                        current_confidence=detection.confidence
                    )
                    events.append(updated_event)

                    self.last_update_times[track_id] = current_timestamp

        # Handle vehicle exits
        events.extend(self._handle_vehicle_exits(current_track_ids, current_timestamp))

        return events

    def _handle_vehicle_exits(self, current_track_ids: set[int], current_timestamp: float) -> list[VehicleEvent]:
        """Handle vehicles that have exited tracking.
        
        Args:
            current_track_ids: Set of currently active track IDs
            current_timestamp: Current timestamp
            
        Returns:
            List of VehicleExited events
        """
        events: list[VehicleEvent] = []
        exited_track_ids = set(self.active_vehicles.keys()) - current_track_ids

        for track_id in exited_track_ids:
            vehicle_state = self.active_vehicles.pop(track_id)
            self.last_update_times.pop(track_id, None)
            self.total_journeys_completed += 1

            # Create complete journey data
            journey = vehicle_state.create_journey(current_timestamp)

            event = VehicleExited(
                track_id=track_id,
                timestamp=current_timestamp,
                vehicle_type=vehicle_state.vehicle_type,
                journey=journey,
                exit_reason="lost_track"  # Could be enhanced with more specific reasons
            )
            events.append(event)

            logger.debug(
                f"Vehicle {track_id} exited: {journey.journey_duration_seconds:.1f}s journey, "
                f"{journey.total_detections} detections, {len(journey.lane_changes)} lane changes"
            )

        return events

    def _handle_empty_frame(self, current_timestamp: float) -> list[VehicleEvent]:
        """Handle empty frame by aging out all active vehicles.
        
        Args:
            current_timestamp: Current timestamp
            
        Returns:
            List of VehicleExited events for all vehicles
        """
        # Update ByteTrack with empty detections
        empty_detections = sv.Detections.empty()
        self.tracker.update_with_detections(empty_detections)

        # Exit all vehicles
        return self._handle_vehicle_exits(set(), current_timestamp)

    def _convert_to_supervision_detections(self, detections: list[DetectionResult]) -> sv.Detections:
        """Convert DetectionResult list to Supervision Detections format."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([
            [det.x1, det.y1, det.x2, det.y2]
            for det in detections
        ], dtype=np.float32)

        confidence = np.array([
            det.confidence
            for det in detections
        ], dtype=np.float32)

        class_id = np.array([
            det.class_id
            for det in detections
        ], dtype=int)

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

    def _convert_from_supervision_detections(self,
                                             sv_detections: sv.Detections,
                                             original_detections: list[DetectionResult]) -> list[DetectionResult]:
        """Convert Supervision Detections back to DetectionResult with track_id."""
        tracked_detections: list[DetectionResult] = []

        if len(sv_detections) == 0 or sv_detections.tracker_id is None:
            return tracked_detections

        # Create working copy for matching
        working_detections: list[DetectionResult | None] = list(original_detections)

        for i, track_id in enumerate(sv_detections.tracker_id):
            tracked_bbox = sv_detections.xyxy[i]

            # Find best matching original detection
            best_match_idx = self._find_best_original_match(tracked_bbox, working_detections)

            if best_match_idx is not None and working_detections[best_match_idx] is not None:
                original = working_detections[best_match_idx]
                assert original is not None

                # Create new DetectionResult with track_id
                tracked_detection = DetectionResult(
                    detection_id=f"track_{track_id}_{original.detection_id}",
                    vehicle_type=original.vehicle_type,
                    confidence=original.confidence,
                    class_id=original.class_id,
                    x1=original.x1,
                    y1=original.y1,
                    x2=original.x2,
                    y2=original.y2,
                    frame_timestamp=original.frame_timestamp,
                    frame_id=original.frame_id,
                    frame_shape=original.frame_shape,
                    track_id=int(track_id)
                )

                tracked_detections.append(tracked_detection)
                working_detections[best_match_idx] = None

        return tracked_detections

    def _find_best_original_match(self,
                                  tracked_bbox: np.ndarray,
                                  original_detections: list[DetectionResult | None]) -> int | None:
        """Find original detection that best matches tracked bbox."""
        best_iou = 0.0
        best_idx = None

        for i, detection in enumerate(original_detections):
            if detection is None:
                continue

            original_bbox = np.array([detection.x1, detection.y1, detection.x2, detection.y2])
            iou = self._calculate_iou(tracked_bbox, original_bbox)

            if iou > best_iou and iou > 0.1:
                best_iou = iou
                best_idx = i

        return best_idx

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area

        return float(intersection_area / union_area) if union_area > 0 else 0.0

    def reset(self) -> None:
        """Reset tracking manager state."""
        self.tracker.reset()
        self.active_vehicles.clear()
        self.last_update_times.clear()
        logger.info("VehicleTrackingManager state reset")

    def get_tracking_stats(self) -> dict[str, Any]:
        """Get comprehensive tracking statistics."""
        return {
            "active_vehicles": len(self.active_vehicles),
            "total_vehicles_tracked": self.total_vehicles_tracked,
            "total_journeys_completed": self.total_journeys_completed,
            "active_track_ids": list(self.active_vehicles.keys())
        }

    def get_vehicle_journey_preview(self, track_id: int) -> dict[str, Any] | None:
        """Get current journey data for an active vehicle (for debugging/monitoring)."""
        if track_id not in self.active_vehicles:
            return None

        vehicle = self.active_vehicles[track_id]
        return {
            "track_id": track_id,
            "vehicle_type": vehicle.vehicle_type.value,
            "duration_so_far": vehicle.journey_duration_seconds,
            "total_detections": vehicle.total_detections,
            "lane_changes": len(vehicle.lane_changes),
            "current_lane": vehicle.current_lane,
            "best_confidence": vehicle.best_confidence,
            "average_confidence": vehicle.average_confidence
        }
