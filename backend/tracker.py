"""Object tracking module for TrafficMetry using Supervision ByteTrack.

This module provides state-of-the-art vehicle tracking capabilities using
ByteTrack algorithm from Roboflow Supervision library for consistent vehicle IDs.
"""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv  # type: ignore

from backend.detection_models import DetectionResult

logger = logging.getLogger(__name__)


class TrackedVehicle:
    """Lightweight tracked vehicle info for statistics and best shot tracking."""

    def __init__(self, track_id: int, detection: DetectionResult):
        """Initialize tracked vehicle.

        Args:
            track_id: Unique tracking ID assigned by ByteTrack
            detection: Initial detection result
        """
        self.track_id = track_id
        self.vehicle_type = detection.vehicle_type
        self.confidence_history: list[float] = [detection.confidence]
        self.bbox_history: list[tuple[int, int, int, int]] = [
            (detection.x1, detection.y1, detection.x2, detection.y2)
        ]
        self.total_detections = 1
        self.last_detection_id = detection.detection_id

    def update(self, detection: DetectionResult) -> None:
        """Update tracked vehicle with new detection.

        Args:
            detection: New detection result
        """
        self.confidence_history.append(detection.confidence)
        self.bbox_history.append((detection.x1, detection.y1, detection.x2, detection.y2))
        self.total_detections += 1
        self.last_detection_id = detection.detection_id

    @property
    def best_confidence(self) -> float:
        """Return highest confidence from detection history."""
        return max(self.confidence_history)

    @property
    def best_bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box with highest confidence."""
        if not self.confidence_history:
            return self.bbox_history[-1] if self.bbox_history else (0, 0, 0, 0)

        best_idx = self.confidence_history.index(self.best_confidence)
        return self.bbox_history[best_idx]

    @property
    def average_confidence(self) -> float:
        """Return average confidence across all detections."""
        return sum(self.confidence_history) / len(self.confidence_history)


class ObjectTracker:
    """Modern object tracker using Supervision ByteTrack algorithm.

    ByteTrack is a state-of-the-art multi-object tracking algorithm that provides
    superior performance compared to SORT, with built-in Kalman filtering,
    lost track recovery, and optimized association algorithms.
    """

    def __init__(self,
                 track_activation_threshold: float = 0.5,
                 lost_track_buffer: int = 30,
                 minimum_matching_threshold: float = 0.8,
                 frame_rate: int = 30):
        """Initialize ByteTrack-based object tracker.

        Args:
            track_activation_threshold: Detection confidence threshold for track activation
            lost_track_buffer: Number of frames to buffer when a track is lost
            minimum_matching_threshold: Threshold for matching tracks with detections
            frame_rate: Video frame rate for prediction algorithms
        """
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate
            )

            # Statistics and vehicle history for enhanced features
            self.tracked_vehicles: dict[int, TrackedVehicle] = {}
            self.total_tracks_created = 0
            self.active_track_ids: set[int] = set()

            logger.info(
                f"ObjectTracker initialized with ByteTrack: "
                f"track_activation_threshold={track_activation_threshold}, lost_track_buffer={lost_track_buffer}, "
                f"minimum_matching_threshold={minimum_matching_threshold}, frame_rate={frame_rate}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ByteTrack: {e}")
            raise

    def update(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        """Update tracker with new detections and return tracked detections.

        Args:
            detections: List of raw detections from VehicleDetector

        Returns:
            List of detections enriched with consistent track_id
        """
        if not detections:
            # Update tracker with empty detections to handle track aging
            empty_detections = sv.Detections.empty()
            self.tracker.update_with_detections(empty_detections)
            self._cleanup_disappeared_vehicles(set())
            return []

        # Convert our DetectionResult to Supervision Detections format
        sv_detections = self._convert_to_supervision_detections(detections)

        # 🎯 MAGIC: ByteTrack does all the heavy lifting in one line!
        tracked_sv_detections = self.tracker.update_with_detections(sv_detections)

        # Convert back to our format with tracker_id assigned
        tracked_detections = self._convert_from_supervision_detections(
            tracked_sv_detections, detections
        )

        # Update vehicle history for statistics and best shot tracking
        self._update_vehicle_history(tracked_detections)

        logger.debug(
            f"ByteTrack update: {len(detections)} raw → {len(tracked_detections)} tracked, "
            f"active tracks: {len(self.active_track_ids)}"
        )

        return tracked_detections

    def _convert_to_supervision_detections(self, detections: list[DetectionResult]) -> sv.Detections:
        """Convert DetectionResult list to Supervision Detections format.

        Args:
            detections: List of detection results

        Returns:
            Supervision Detections object ready for ByteTrack
        """
        if not detections:
            return sv.Detections.empty()

        # Extract data arrays for supervision
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

    def _convert_from_supervision_detections(
        self,
        sv_detections: sv.Detections,
        original_detections: list[DetectionResult]
    ) -> list[DetectionResult]:
        """Convert Supervision Detections back to DetectionResult with track_id.

        Args:
            sv_detections: Tracked detections from ByteTrack
            original_detections: Original detection list for metadata

        Returns:
            List of DetectionResult objects with track_id assigned
        """
        tracked_detections: list[DetectionResult] = []

        if len(sv_detections) == 0 or sv_detections.tracker_id is None:
            return tracked_detections

        # Create working copy with None support for matching
        working_detections: list[DetectionResult | None] = list(original_detections)

        # Match tracked detections back to originals by bbox similarity
        for i, track_id in enumerate(sv_detections.tracker_id):
            tracked_bbox = sv_detections.xyxy[i]  # [x1, y1, x2, y2]

            # Find best matching original detection
            best_match_idx = self._find_best_original_match(tracked_bbox, working_detections)

            if best_match_idx is not None and working_detections[best_match_idx] is not None:
                original = working_detections[best_match_idx]
                assert original is not None  # Help MyPy understand type narrowing

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
                # Mark as matched to avoid double matching
                working_detections[best_match_idx] = None

        return tracked_detections

    def _find_best_original_match(
        self,
        tracked_bbox: np.ndarray,
        original_detections: list[DetectionResult | None]
    ) -> int | None:
        """Find original detection that best matches tracked bbox.

        Args:
            tracked_bbox: Tracked bounding box [x1, y1, x2, y2]
            original_detections: List of original detections

        Returns:
            Index of best matching detection, or None if no good match
        """
        best_iou = 0.0
        best_idx = None

        for i, detection in enumerate(original_detections):
            if detection is None:  # Already matched
                continue

            # Calculate IoU between tracked bbox and original detection
            original_bbox = np.array([detection.x1, detection.y1, detection.x2, detection.y2])
            iou = self._calculate_iou(tracked_bbox, original_bbox)

            if iou > best_iou and iou > 0.1:  # Minimum IoU threshold
                best_iou = iou
                best_idx = i

        return best_idx

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area <= 0:
            return 0.0

        return float(intersection_area / union_area)

    def _update_vehicle_history(self, tracked_detections: list[DetectionResult]) -> None:
        """Update tracked vehicle history with new detections.

        Args:
            tracked_detections: List of detections with track_id
        """
        current_track_ids = set()

        for detection in tracked_detections:
            if not hasattr(detection, 'track_id') or detection.track_id is None:
                continue

            track_id = detection.track_id
            current_track_ids.add(track_id)

            if track_id not in self.tracked_vehicles:
                # New vehicle track discovered
                self.tracked_vehicles[track_id] = TrackedVehicle(track_id, detection)
                self.total_tracks_created += 1
                logger.debug(f"New vehicle track: ID={track_id}, type={detection.vehicle_type}")
            else:
                # Update existing track
                self.tracked_vehicles[track_id].update(detection)

        # Update active tracks and cleanup
        self.active_track_ids = current_track_ids
        self._cleanup_disappeared_vehicles(current_track_ids)

    def _cleanup_disappeared_vehicles(self, current_track_ids: set[int]) -> None:
        """Clean up vehicles that are no longer being tracked.

        Args:
            current_track_ids: Set of currently active track IDs
        """
        # Find disappeared vehicles
        disappeared_ids = set(self.tracked_vehicles.keys()) - current_track_ids

        for track_id in disappeared_ids:
            vehicle = self.tracked_vehicles.pop(track_id, None)
            if vehicle:
                logger.debug(
                    f"Vehicle track ended: ID={track_id}, "
                    f"detections={vehicle.total_detections}, "
                    f"best_confidence={vehicle.best_confidence:.2f}"
                )

    def get_vehicle_best_detection(self, track_id: int) -> DetectionResult | None:
        """Get the best detection for a specific tracked vehicle.

        Args:
            track_id: Tracking ID of the vehicle

        Returns:
            Best DetectionResult for the vehicle, or None if not found
        """
        if track_id not in self.tracked_vehicles:
            return None

        vehicle = self.tracked_vehicles[track_id]
        x1, y1, x2, y2 = vehicle.best_bbox

        # Create DetectionResult from best detection
        return DetectionResult(
            detection_id=f"best_{track_id}",
            vehicle_type=vehicle.vehicle_type,
            confidence=vehicle.best_confidence,
            class_id=0,  # Not relevant for best detection
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            frame_timestamp=0,  # Timestamp not relevant for best detection
            frame_id=0,  # Not relevant for best detection
            frame_shape=(0, 0, 0),  # Not relevant for best detection
            track_id=track_id
        )

    def reset(self) -> None:
        """Reset tracker state - useful between video processing sessions."""
        self.tracker.reset()
        self.tracked_vehicles.clear()
        self.active_track_ids.clear()
        logger.info("ObjectTracker state reset")

    def get_tracking_stats(self) -> dict[str, int]:
        """Get comprehensive tracking statistics.

        Returns:
            Dictionary with tracking statistics
        """
        return {
            "active_tracks": len(self.active_track_ids),
            "total_tracks_created": self.total_tracks_created,
            "tracked_vehicles_in_memory": len(self.tracked_vehicles)
        }
