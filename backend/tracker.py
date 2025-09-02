"""Vehicle tracking module with event-driven lifecycle management.

This module provides state-of-the-art vehicle tracking using ByteTrack algorithm
from Roboflow Supervision, enhanced with complete vehicle journey lifecycle
management, dynamic direction detection, and event-driven architecture.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import supervision as sv

from backend.detection_models import DetectionResult
from backend.direction_analyzer import DynamicDirectionAnalyzer, MovementAnalytics
from backend.journey_manager import JourneyIDManager
from backend.vehicle_events import (
    VehicleEntered,
    VehicleEvent,
    VehicleExited,
    VehicleJourney,
    VehicleUpdated,
)

logger = logging.getLogger(__name__)


class VehicleTrackingState:
    """Enhanced state tracking for a single vehicle journey with dynamic direction detection.

    Tracks essential data and position history for dynamic movement analysis
    without static lane dependencies.
    """

    MAX_POSITION_HISTORY = 15  # Keep last 15 positions (~0.5s at 30fps)

    def __init__(self, track_id: int, detection: DetectionResult, journey_id: str):
        """Initialize vehicle tracking state.

        Args:
            track_id: Unique tracking ID assigned by ByteTrack
            detection: Initial detection result
            journey_id: Global unique journey identifier
        """
        self.track_id = track_id
        self.journey_id = journey_id
        self.vehicle_type = detection.vehicle_type

        # Essential timestamps
        self.entry_timestamp = detection.frame_timestamp
        self.last_update_timestamp = detection.frame_timestamp

        # Position tracking for dynamic analysis
        self.position_history: List[Tuple[float, Tuple[int, int]]] = [
            (detection.frame_timestamp, detection.centroid)
        ]
        self.entry_position = detection.centroid
        self.current_position = detection.centroid

        # Dynamic direction detection
        self.movement_direction: Optional[str] = None  # "left", "right", "stationary"
        self.direction_confidence: float = 0.0
        self.direction_analyzer = DynamicDirectionAnalyzer()

        # Essential metrics
        self.total_detections = 1
        self.best_detection = detection
        self.best_confidence = detection.confidence

    def update(self, detection: DetectionResult) -> None:
        """Update vehicle state with new detection and analyze movement.

        Args:
            detection: New detection result
        """
        # Update essential metrics
        self.total_detections += 1
        self.last_update_timestamp = detection.frame_timestamp
        self.current_position = detection.centroid

        # Update best detection if confidence improved
        if detection.confidence > self.best_confidence:
            self.best_detection = detection
            self.best_confidence = detection.confidence

        # Add to position history with memory management
        self.position_history.append((detection.frame_timestamp, detection.centroid))
        if len(self.position_history) > self.MAX_POSITION_HISTORY:
            self.position_history.pop(0)  # Remove oldest position

        # Update dynamic direction analysis
        self._update_movement_direction()

    def _update_movement_direction(self) -> None:
        """Update movement direction based on position history."""
        new_direction, new_confidence = self.direction_analyzer.analyze_movement_direction(
            self.position_history
        )

        # Smooth direction transitions
        smoothed_direction, smoothed_confidence = self.direction_analyzer.smooth_direction_transition(
            self.movement_direction, new_direction, new_confidence, self.direction_confidence
        )

        self.movement_direction = smoothed_direction
        self.direction_confidence = smoothed_confidence

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
            Complete VehicleJourney data with movement analytics
        """
        # Calculate movement analytics
        total_distance = MovementAnalytics.calculate_total_distance(self.position_history)
        average_speed = MovementAnalytics.calculate_average_speed(self.position_history)
        displacement_vector = MovementAnalytics.get_displacement_vector(self.position_history)
        
        return VehicleJourney(
            track_id=self.track_id,
            journey_id=self.journey_id,
            vehicle_type=self.vehicle_type,
            entry_timestamp=self.entry_timestamp,
            exit_timestamp=exit_timestamp,
            journey_duration_seconds=exit_timestamp - self.entry_timestamp,
            
            # Position-based data (replaces lane data)
            entry_position=self.entry_position,
            exit_position=self.current_position,
            movement_direction=self.movement_direction,
            direction_confidence=self.direction_confidence,
            
            # Movement analytics  
            total_movement_pixels=total_distance,
            average_speed_pixels_per_second=average_speed,
            displacement_vector=displacement_vector,
            
            # Essential metrics
            total_detections=self.total_detections,
            best_confidence=self.best_confidence,
            best_bbox=self.best_bbox,
            best_detection_timestamp=self.best_detection.frame_timestamp
        )


class VehicleTrackingManager:
    """Advanced vehicle tracking with dynamic direction detection and unique journey IDs.
    
    Event-driven architecture that manages vehicle journeys from entry to exit,
    providing real-time updates, dynamic movement analysis, and complete journey analytics
    without static lane dependencies.
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

            # Journey ID management
            self.journey_id_manager = JourneyIDManager()

            # Statistics
            self.total_vehicles_tracked = 0
            self.total_journeys_completed = 0

            logger.info(
                f"VehicleTrackingManager initialized with dynamic direction detection: "
                f"activation_threshold={track_activation_threshold}, "
                f"lost_buffer={lost_track_buffer}, "
                f"matching_threshold={minimum_matching_threshold}, "
                f"update_interval={update_interval_seconds}s"
            )

        except Exception as e:
            logger.error(f"Failed to initialize VehicleTrackingManager: {e}")
            raise

    def update(self, detections: list[DetectionResult]) -> tuple[list[DetectionResult], list[VehicleEvent]]:
        """Update tracking and generate vehicle lifecycle events with dynamic direction detection.

        Args:
            detections: List of raw detections from VehicleDetector

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
            tracked_detections, current_timestamp
        ))

        return tracked_detections, events

    def _generate_lifecycle_events(self,
                                   tracked_detections: list[DetectionResult],
                                   current_timestamp: float) -> list[VehicleEvent]:
        """Generate vehicle lifecycle events from tracked detections with dynamic analysis.
        
        Args:
            tracked_detections: Detections with track_id assigned
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

            if track_id not in self.active_vehicles:
                # New vehicle entered - create unique journey ID
                journey_id = self.journey_id_manager.create_journey_id(track_id)
                vehicle_state = VehicleTrackingState(track_id, detection, journey_id)
                self.active_vehicles[track_id] = vehicle_state
                self.total_vehicles_tracked += 1

                entered_event = VehicleEntered(
                    track_id=track_id,
                    journey_id=journey_id,
                    timestamp=detection.frame_timestamp,
                    vehicle_type=detection.vehicle_type,
                    detection=detection
                )
                events.append(entered_event)

                # Set initial update time
                self.last_update_times[track_id] = current_timestamp

                logger.info(f"🚗 Vehicle {journey_id} (Track {track_id}) entered: {detection.vehicle_type}")

            else:
                # Update existing vehicle with dynamic direction analysis
                vehicle_state = self.active_vehicles[track_id]
                vehicle_state.update(detection)

                # Generate VehicleUpdated event if enough time has passed
                last_update = self.last_update_times.get(track_id, 0)
                if current_timestamp - last_update >= self.update_interval_seconds:

                    updated_event = VehicleUpdated(
                        track_id=track_id,
                        journey_id=vehicle_state.journey_id,
                        timestamp=detection.frame_timestamp,
                        vehicle_type=detection.vehicle_type,
                        detection=detection,
                        movement_direction=vehicle_state.movement_direction,
                        direction_confidence=vehicle_state.direction_confidence,
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

            # Release journey ID
            journey_id = self.journey_id_manager.release_journey_id(track_id)

            # Create complete journey data with analytics
            journey = vehicle_state.create_journey(current_timestamp)

            event = VehicleExited(
                track_id=track_id,
                journey_id=journey_id or vehicle_state.journey_id,
                timestamp=current_timestamp,
                vehicle_type=vehicle_state.vehicle_type,
                journey=journey,
                exit_reason="lost_track"  # Could be enhanced with more specific reasons
            )
            events.append(event)

            logger.info(
                f"🏁 Vehicle {journey_id} (Track {track_id}) journey completed: "
                f"{journey.journey_duration_seconds:.1f}s, {journey.total_detections} detections, "
                f"direction: {journey.movement_direction} ({journey.direction_confidence:.2f}), "
                f"distance: {journey.total_movement_pixels:.0f}px"
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
        """Convert Supervision Detections back to DetectionResult with track_id.
        
        Simple 1:1 mapping approach - Trust ByteTrack 100% for correct ordering.
        """
        tracked_detections: list[DetectionResult] = []

        if len(sv_detections) == 0 or sv_detections.tracker_id is None:
            return tracked_detections

        # Trust ByteTrack: Simple 1:1 mapping by index
        for i, track_id in enumerate(sv_detections.tracker_id):
            if i < len(original_detections):
                original = original_detections[i]
                
                # Create DetectionResult with ByteTrack's track_id
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
            else:
                logger.warning(f"ByteTrack returned more tracks ({len(sv_detections.tracker_id)}) than original detections ({len(original_detections)})")

        return tracked_detections


    def reset(self) -> None:
        """Reset tracking manager state."""
        self.tracker.reset()
        self.active_vehicles.clear()
        self.last_update_times.clear()
        self.journey_id_manager.reset()
        logger.info("VehicleTrackingManager state reset")

    def get_tracking_stats(self) -> dict[str, Any]:
        """Get comprehensive tracking statistics including journey management."""
        journey_stats = self.journey_id_manager.get_statistics()
        
        return {
            "active_vehicles": len(self.active_vehicles),
            "total_vehicles_tracked": self.total_vehicles_tracked,
            "total_journeys_completed": self.total_journeys_completed,
            "active_track_ids": list(self.active_vehicles.keys()),
            "active_journey_ids": self.journey_id_manager.get_active_journey_ids(),
            "journey_statistics": journey_stats
        }

    def get_vehicle_journey_preview(self, track_id: int) -> dict[str, Any] | None:
        """Get current journey data for an active vehicle (for debugging/monitoring)."""
        if track_id not in self.active_vehicles:
            return None

        vehicle = self.active_vehicles[track_id]
        return {
            "track_id": track_id,
            "journey_id": vehicle.journey_id,
            "vehicle_type": vehicle.vehicle_type.value,
            "duration_so_far": vehicle.journey_duration_seconds,
            "total_detections": vehicle.total_detections,
            "current_position": vehicle.current_position,
            "entry_position": vehicle.entry_position,
            "movement_direction": vehicle.movement_direction,
            "direction_confidence": vehicle.direction_confidence,
            "best_confidence": vehicle.best_confidence,
            "position_history_length": len(vehicle.position_history)
        }
    
    def cleanup_stale_journeys(self) -> int:
        """Cleanup stale journey IDs that no longer have active vehicles."""
        active_track_ids = set(self.active_vehicles.keys())
        return self.journey_id_manager.cleanup_stale_journeys(active_track_ids)
