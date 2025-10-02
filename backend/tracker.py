"""Vehicle tracking module with event-driven lifecycle management.

This module provides state-of-the-art vehicle tracking using ByteTrack algorithm
from Roboflow Supervision, enhanced with complete vehicle journey lifecycle
management, dynamic direction detection, and event-driven architecture.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import supervision as sv

from backend.config import ModelSettings
from backend.detection_models import DetectionResult
from backend.direction_analyzer import DynamicDirectionAnalyzer
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
        self.position_history: list[tuple[float, tuple[int, int]]] = [
            (detection.frame_timestamp, detection.centroid)
        ]
        self.entry_pos_x, self.entry_pos_y = detection.centroid
        self.current_pos_x, self.current_pos_y = detection.centroid

        # Dynamic direction detection
        self.movement_direction: str | None = None  # "left", "right", "stationary"
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
        self.current_pos_x, self.current_pos_y = detection.centroid

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
        smoothed_direction, _ = self.direction_analyzer.smooth_direction_transition(
            self.movement_direction, new_direction, new_confidence, 0.0
        )

        self.movement_direction = smoothed_direction

    @property
    def journey_duration_seconds(self) -> float:
        """Calculate journey duration in seconds."""
        return self.last_update_timestamp - self.entry_timestamp

    @property
    def best_bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box of best detection."""
        return (
            self.best_detection.x1,
            self.best_detection.y1,
            self.best_detection.x2,
            self.best_detection.y2,
        )

    def create_journey(self, exit_timestamp: float) -> VehicleJourney:
        """Create complete journey data for database storage.

        Args:
            exit_timestamp: When the vehicle exited tracking

        Returns:
            Complete VehicleJourney data with clean position tracking
        """
        return VehicleJourney(
            journey_id=self.journey_id,
            vehicle_type=self.vehicle_type,
            entry_timestamp=self.entry_timestamp,
            exit_timestamp=exit_timestamp,
            journey_duration_seconds=exit_timestamp - self.entry_timestamp,
            # Clean position data (dedicated fields)
            entry_pos_x=self.entry_pos_x,
            entry_pos_y=self.entry_pos_y,
            exit_pos_x=self.current_pos_x,
            exit_pos_y=self.current_pos_y,
            movement_direction=self.movement_direction,
            # Essential metrics
            total_detections=self.total_detections,
            best_confidence=self.best_confidence,
            best_bbox=self.best_bbox,
            best_detection_timestamp=self.best_detection.frame_timestamp,
        )


class VehicleTrackingManager:
    """Advanced vehicle tracking with dynamic direction detection and unique journey IDs.

    Event-driven architecture that manages vehicle journeys from entry to exit,
    providing real-time updates, dynamic movement analysis, and complete journey analytics
    without static lane dependencies.
    """

    def __init__(
        self,
        config: ModelSettings,
        frame_rate: int = 30,
        update_interval_seconds: float = 1.0,
        start_journey_counter: int = 0,
    ):
        """Initialize vehicle tracking manager with full config integration.

        Args:
            config: Model settings with ALL ByteTrack parameters from config
            frame_rate: Video frame rate for prediction algorithms
            update_interval_seconds: Minimum interval between VehicleUpdated events
            start_journey_counter: Starting counter for journey ID continuation
        """
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=config.track_activation_threshold,  # From config
                lost_track_buffer=config.lost_track_buffer,                   # From config
                minimum_matching_threshold=config.minimum_matching_threshold,  # From config
                frame_rate=frame_rate,
                minimum_consecutive_frames=config.minimum_consecutive_frames,  # From config
            )

            # Store config for memory management
            self.config = config

            # Vehicle lifecycle management
            self.active_vehicles: dict[int, VehicleTrackingState] = {}
            self.update_interval_seconds = update_interval_seconds
            self.last_update_times: dict[int, float] = {}

            # Memory management parameters
            self.max_active_vehicles = 100  # Default max active vehicles
            self.max_history_per_vehicle = 30  # Max 30 positions in history
            self.memory_cleanup_interval = 100  # Every 100 updates
            self.memory_cleanup_counter = 0

            # Journey ID management with continuation support
            self.journey_id_manager = JourneyIDManager(start_counter=start_journey_counter)

            # Track confirmation system - uses config parameter
            self.pending_tracks: dict[int, list[DetectionResult]] = {}  # track_id -> detections
            self.track_confirmation_threshold = config.minimum_consecutive_frames

            # Event broadcasting system
            self.event_listeners: list[Callable[[VehicleEvent, np.ndarray], None]] = []

            # Statistics
            self.total_vehicles_tracked = 0
            self.total_journeys_completed = 0

            logger.info(
                f"VehicleTrackingManager initialized with configurable ByteTrack: "
                f"activation_threshold={config.track_activation_threshold}, "
                f"lost_buffer={config.lost_track_buffer}, "
                f"matching_threshold={config.minimum_matching_threshold}, "
                f"update_interval={update_interval_seconds}s, "
                f"track_confirmation_threshold={config.minimum_consecutive_frames} frames"
            )

        except Exception as e:
            logger.error(f"Failed to initialize VehicleTrackingManager: {e}")
            raise

    def add_event_listener(self, listener: Callable[[VehicleEvent, np.ndarray], None]) -> None:
        """Add event listener for vehicle events.

        Args:
            listener: Function that handles (event, frame) pairs
        """
        self.event_listeners.append(listener)
        logger.info(
            f"Added event listener: {listener.__name__ if hasattr(listener, '__name__') else type(listener).__name__}"
        )

    def _broadcast_event(self, event: VehicleEvent, frame: np.ndarray) -> None:
        """Broadcast event to all registered listeners.

        Args:
            event: Vehicle event to broadcast
            frame: Current video frame
        """
        for listener in self.event_listeners:
            try:
                listener(event, frame)
            except Exception as e:
                logger.error(f"Event listener error: {e}")

    def update(
        self, detections: list[DetectionResult], current_frame: np.ndarray
    ) -> tuple[list[DetectionResult], list[VehicleEvent]]:
        """Update tracking with track confirmation and event broadcasting.

        Args:
            detections: List of raw detections from VehicleDetector
            current_frame: Current video frame for event broadcasting

        Returns:
            Tuple of (tracked_detections, vehicle_events)
        """
        current_timestamp = time.time()
        events: list[VehicleEvent] = []

        if not detections:
            # Handle empty frame - check for vehicle exits
            events.extend(self._handle_empty_frame(current_timestamp, current_frame))
            return [], events

        # Convert detections to supervision format
        sv_detections = self._convert_to_supervision_detections(detections)

        # Run ByteTrack tracking
        tracked_sv_detections = self.tracker.update_with_detections(sv_detections)

        # Convert back to DetectionResult with track_id
        tracked_detections = self._convert_from_supervision_detections(
            tracked_sv_detections, detections
        )

        # Generate lifecycle events with track confirmation and broadcasting
        events.extend(
            self._generate_lifecycle_events(tracked_detections, current_timestamp, current_frame)
        )

        # Periodic memory cleanup for performance optimization
        if self._should_cleanup_memory():
            cleanup_stats = self._cleanup_memory()
            self.memory_cleanup_counter = 0  # Reset counter

            if any(cleanup_stats.values()):  # Log only if something was cleaned
                logger.debug(
                    f"🧹 Memory cleanup completed: "
                    f"history_trimmed={cleanup_stats['history_trimmed']}, "
                    f"vehicles_forced_exit={cleanup_stats['vehicles_forced_exit']}, "
                    f"pending_tracks_cleaned={cleanup_stats['pending_tracks_cleaned']}"
                )

        return tracked_detections, events

    def _generate_lifecycle_events(
        self,
        tracked_detections: list[DetectionResult],
        current_timestamp: float,
        current_frame: np.ndarray,
    ) -> list[VehicleEvent]:
        """Generate vehicle lifecycle events with track confirmation and broadcasting.

        Args:
            tracked_detections: Detections with track_id assigned
            current_timestamp: Current frame timestamp
            current_frame: Current video frame for broadcasting

        Returns:
            List of vehicle events that occurred
        """
        events: list[VehicleEvent] = []
        current_track_ids = set()

        # Process all detections with track confirmation
        for detection in tracked_detections:
            if detection.track_id is None:
                continue

            track_id = detection.track_id
            current_track_ids.add(track_id)

            if track_id not in self.active_vehicles:
                # TRACK CONFIRMATION LOGIC: Don't create journey immediately

                if track_id not in self.pending_tracks:
                    # First time seeing this track_id
                    self.pending_tracks[track_id] = []
                    logger.debug(f"🕐 New track {track_id} pending confirmation")

                # Add detection to pending list
                self.pending_tracks[track_id].append(detection)

                # Check if track is confirmed (enough consecutive detections)
                if len(self.pending_tracks[track_id]) >= self.track_confirmation_threshold:
                    # TRACK CONFIRMED - create journey now!
                    journey_id = self.journey_id_manager.create_journey_id(track_id)

                    # Use FIRST detection as entry point
                    first_detection = self.pending_tracks[track_id][0]
                    vehicle_state = VehicleTrackingState(track_id, first_detection, journey_id)

                    # Apply all pending detections to catch up
                    for pending_detection in self.pending_tracks[track_id][1:]:
                        vehicle_state.update(pending_detection)

                    # Now add current detection
                    vehicle_state.update(detection)

                    self.active_vehicles[track_id] = vehicle_state
                    self.total_vehicles_tracked += 1

                    # Generate VehicleEntered event (with first detection timestamp)
                    entered_event = VehicleEntered(
                        track_id=track_id,
                        journey_id=journey_id,
                        timestamp=first_detection.frame_timestamp,
                        vehicle_type=first_detection.vehicle_type,
                        detection=first_detection,
                    )
                    events.append(entered_event)

                    # Set initial update time
                    self.last_update_times[track_id] = current_timestamp

                    # Broadcast VehicleEntered event
                    self._broadcast_event(entered_event, current_frame)

                    # Clean up pending
                    del self.pending_tracks[track_id]

                    logger.info(
                        f"✅ Track {track_id} confirmed -> {journey_id} after {self.track_confirmation_threshold} detections"
                    )

                else:
                    # Still pending - skip further processing for this detection
                    continue

            else:
                # Update existing confirmed vehicle
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
                        total_detections_so_far=vehicle_state.total_detections,
                        current_confidence=detection.confidence,
                    )
                    events.append(updated_event)
                    self.last_update_times[track_id] = current_timestamp

                    # Broadcast VehicleUpdated event
                    self._broadcast_event(updated_event, current_frame)

        # Handle vehicle exits
        events.extend(
            self._handle_vehicle_exits(current_track_ids, current_timestamp, current_frame)
        )

        return events

    def _handle_vehicle_exits(
        self, current_track_ids: set[int], current_timestamp: float, current_frame: np.ndarray
    ) -> list[VehicleEvent]:
        """Handle vehicles that have exited tracking.

        Args:
            current_track_ids: Set of currently active track IDs
            current_timestamp: Current timestamp
            current_frame: Current video frame for event broadcasting

        Returns:
            List of VehicleExited events
        """
        events: list[VehicleEvent] = []
        exited_track_ids = set(self.active_vehicles.keys()) - current_track_ids

        for track_id in exited_track_ids:
            vehicle_state = self.active_vehicles.pop(track_id)
            self.last_update_times.pop(track_id, None)

            # ALL CONFIRMED JOURNEYS ARE VALID - no filtering needed!
            self.total_journeys_completed += 1

            # Release journey ID
            journey_id = self.journey_id_manager.release_journey_id(track_id)

            # Create complete journey data
            journey = vehicle_state.create_journey(current_timestamp)

            event = VehicleExited(
                track_id=track_id,
                journey_id=journey_id or vehicle_state.journey_id,
                timestamp=current_timestamp,
                vehicle_type=vehicle_state.vehicle_type,
                journey=journey,
                exit_reason="lost_track",
            )
            events.append(event)

            # Broadcast VehicleExited event
            self._broadcast_event(event, current_frame)

            logger.info(
                f"🏁 Confirmed vehicle {journey_id} (Track {track_id}) completed: "
                f"{journey.journey_duration_seconds:.1f}s, {journey.total_detections} detections, "
                f"direction: {journey.movement_direction}"
            )

        return events

    def _handle_empty_frame(
        self, current_timestamp: float, current_frame: np.ndarray
    ) -> list[VehicleEvent]:
        """Handle empty frame by aging out all active vehicles.

        Args:
            current_timestamp: Current timestamp
            current_frame: Current video frame for event broadcasting

        Returns:
            List of VehicleExited events for all vehicles
        """
        # Update ByteTrack with empty detections
        empty_detections = sv.Detections.empty()
        self.tracker.update_with_detections(empty_detections)

        # Exit all vehicles
        return self._handle_vehicle_exits(set(), current_timestamp, current_frame)

    def _convert_to_supervision_detections(
        self, detections: list[DetectionResult]
    ) -> sv.Detections:
        """Convert DetectionResult list to Supervision Detections format."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([[det.x1, det.y1, det.x2, det.y2] for det in detections], dtype=np.float32)

        confidence = np.array([det.confidence for det in detections], dtype=np.float32)

        class_id = np.array([det.class_id for det in detections], dtype=int)

        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def _convert_from_supervision_detections(
        self, sv_detections: sv.Detections, original_detections: list[DetectionResult]
    ) -> list[DetectionResult]:
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
                    track_id=int(track_id),
                )
                tracked_detections.append(tracked_detection)
            else:
                logger.warning(
                    f"ByteTrack returned more tracks ({len(sv_detections.tracker_id)}) than original detections ({len(original_detections)})"
                )

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
            "track_confirmation_threshold": self.track_confirmation_threshold,
            "data_quality_ratio": f"{self.total_journeys_completed}/{self.total_vehicles_tracked}"
            if self.total_vehicles_tracked > 0
            else "0/0",
            "active_track_ids": list(self.active_vehicles.keys()),
            "active_journey_ids": self.journey_id_manager.get_active_journey_ids(),
            "journey_statistics": journey_stats,
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
            "current_position": (vehicle.current_pos_x, vehicle.current_pos_y),
            "entry_position": (vehicle.entry_pos_x, vehicle.entry_pos_y),
            "movement_direction": vehicle.movement_direction,
            "best_confidence": vehicle.best_confidence,
            "position_history_length": len(vehicle.position_history),
        }

    def cleanup_stale_journeys(self) -> int:
        """Cleanup stale journey IDs that no longer have active vehicles."""
        active_track_ids = set(self.active_vehicles.keys())
        return self.journey_id_manager.cleanup_stale_journeys(active_track_ids)

    def _cleanup_memory(self) -> dict[str, int]:
        """Periodic memory cleanup for performance optimization.

        Returns:
            Dictionary with cleanup operation statistics
        """
        cleanup_stats = {
            "history_trimmed": 0,
            "vehicles_forced_exit": 0,
            "pending_tracks_cleaned": 0
        }

        # 1. Limit position history for active vehicles
        for vehicle in self.active_vehicles.values():
            if len(vehicle.position_history) > self.max_history_per_vehicle:
                vehicle.position_history = vehicle.position_history[-self.max_history_per_vehicle:]
                cleanup_stats["history_trimmed"] += 1

        # 2. Limit number of active vehicles (remove oldest ones)
        if len(self.active_vehicles) > self.max_active_vehicles:
            # Sort vehicles by last update timestamp
            sorted_vehicles = sorted(
                self.active_vehicles.items(),
                key=lambda x: x[1].last_update_timestamp
            )

            # Remove oldest vehicles exceeding the limit
            excess_count = len(self.active_vehicles) - self.max_active_vehicles
            for track_id, _vehicle in sorted_vehicles[:excess_count]:
                self._force_exit_vehicle(track_id, "memory_cleanup")
                cleanup_stats["vehicles_forced_exit"] += 1

        # 3. Clean up excessively long pending tracks
        max_pending_age = 10  # Max 10 detections in pending
        tracks_to_remove = []
        for track_id, detections in self.pending_tracks.items():
            if len(detections) > max_pending_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.pending_tracks[track_id]
            cleanup_stats["pending_tracks_cleaned"] += 1

        return cleanup_stats

    def _force_exit_vehicle(self, track_id: int, reason: str) -> None:
        """Force vehicle journey termination for memory management.

        Args:
            track_id: Track ID to terminate
            reason: Reason for termination (e.g. "memory_cleanup")
        """
        if track_id not in self.active_vehicles:
            return

        vehicle_state = self.active_vehicles.pop(track_id)
        self.last_update_times.pop(track_id, None)

        # Increment completed journeys counter
        self.total_journeys_completed += 1

        # Release journey ID
        journey_id = self.journey_id_manager.release_journey_id(track_id)

        logger.info(
            f"🧹 Forced exit for memory cleanup: {journey_id or vehicle_state.journey_id} "
            f"(Track {track_id}) - reason: {reason}, "
            f"duration: {vehicle_state.journey_duration_seconds:.1f}s, "
            f"detections: {vehicle_state.total_detections}"
        )

    def _should_cleanup_memory(self) -> bool:
        """Check if memory cleanup should be performed.

        Returns:
            True if cleanup should be performed
        """
        self.memory_cleanup_counter += 1

        # Regular cleanup every X updates or when too many active vehicles
        return (
            self.memory_cleanup_counter >= self.memory_cleanup_interval or
            len(self.active_vehicles) > self.max_active_vehicles * 1.2  # 20% over limit
        )
