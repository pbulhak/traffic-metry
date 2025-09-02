"""Event-driven candidate saver that saves only the best image per confirmed journey."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.detection_models import DetectionResult, VehicleType
from backend.vehicle_events import VehicleEntered, VehicleExited, VehicleUpdated

logger = logging.getLogger(__name__)


@dataclass
class BestCandidate:
    """Tracks the best candidate for a specific journey."""

    journey_id: str
    track_id: int
    vehicle_type: VehicleType
    entry_timestamp: float

    # Best detection data
    best_detection: DetectionResult
    best_confidence: float
    best_frame: np.ndarray | None = None  # Store the actual frame data

    def update_if_better(self, detection: DetectionResult, frame: np.ndarray) -> bool:
        """Update if this detection has higher confidence.

        Args:
            detection: New detection result
            frame: Current video frame

        Returns:
            True if this detection was better and updated
        """
        if detection.confidence > self.best_confidence:
            self.best_detection = detection
            self.best_confidence = detection.confidence
            self.best_frame = frame.copy()  # Store frame for later saving
            logger.debug(
                f"📈 Journey {self.journey_id}: new best confidence {detection.confidence:.3f}"
            )
            return True
        return False


class EventDrivenCandidateSaver:
    """Event-driven candidate saver that saves only the best image per confirmed journey."""

    def __init__(self, output_dir: Path, storage_limit_gb: float = 10.0):
        """Initialize event-driven candidate saver.

        Args:
            output_dir: Directory to save candidate images
            storage_limit_gb: Storage limit in gigabytes
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage management
        self.storage_limit_bytes = int(storage_limit_gb * 1024**3)
        self.saved_count = 0

        # Event-driven architecture: Track best candidates per journey
        self.active_journeys: dict[str, BestCandidate] = {}  # journey_id -> best candidate

        logger.info(f"EventDrivenCandidateSaver initialized - output: {output_dir}")
        logger.info("Strategy: ONE best image per confirmed journey")
        logger.info(
            f"Storage limit: {storage_limit_gb:.1f} GB ({self.storage_limit_bytes:,} bytes)"
        )

    def handle_vehicle_entered(self, event: VehicleEntered, frame: np.ndarray) -> None:
        """Start tracking candidates for newly confirmed journey.

        Args:
            event: VehicleEntered event from confirmed track
            frame: Current video frame
        """
        journey_id = event.journey_id

        # Initialize best candidate with entry detection
        best_candidate = BestCandidate(
            journey_id=journey_id,
            track_id=event.track_id,
            vehicle_type=event.vehicle_type,
            entry_timestamp=event.timestamp,
            best_detection=event.detection,
            best_confidence=event.detection.confidence,
            best_frame=frame.copy(),
        )

        self.active_journeys[journey_id] = best_candidate
        logger.info(
            f"🎯 Started tracking candidates for {journey_id} (confidence: {event.detection.confidence:.3f})"
        )

    def handle_vehicle_updated(self, event: VehicleUpdated, frame: np.ndarray) -> None:
        """Update best candidate if current detection is better.

        Args:
            event: VehicleUpdated event from active journey
            frame: Current video frame
        """
        journey_id = event.journey_id

        if journey_id in self.active_journeys:
            candidate = self.active_journeys[journey_id]
            candidate.update_if_better(event.detection, frame)
        else:
            logger.warning(f"Received update for unknown journey {journey_id}")

    def handle_vehicle_exited(self, event: VehicleExited) -> Path | None:
        """Save the best candidate image for completed journey.

        Args:
            event: VehicleExited event from completed journey

        Returns:
            Path to saved image or None if saving failed
        """
        journey_id = event.journey_id

        if journey_id not in self.active_journeys:
            logger.warning(f"No candidate data for exiting journey {journey_id}")
            return None

        candidate = self.active_journeys.pop(journey_id)

        # Save the best image
        saved_path = self._save_best_candidate(candidate, event)

        if saved_path:
            self.saved_count += 1
            logger.info(
                f"💾 Saved BEST candidate for {journey_id}: "
                f"confidence={candidate.best_confidence:.3f}, path={saved_path.name}"
            )

        return saved_path

    def _save_best_candidate(
        self, candidate: BestCandidate, exit_event: VehicleExited
    ) -> Path | None:
        """Save the best candidate image to disk.

        Args:
            candidate: Best candidate data for the journey
            exit_event: VehicleExited event with journey data

        Returns:
            Path to saved image or None if saving failed
        """
        try:
            if candidate.best_frame is None:
                logger.error(f"No frame data for {candidate.journey_id}")
                return None

            # Extract vehicle crop from best frame
            detection = candidate.best_detection
            frame = candidate.best_frame

            # Bounds validation - ensure coordinates are within frame dimensions
            frame_height, frame_width = frame.shape[:2]
            x1 = max(0, min(detection.x1, frame_width - 1))
            y1 = max(0, min(detection.y1, frame_height - 1))
            x2 = max(x1 + 1, min(detection.x2, frame_width))
            y2 = max(y1 + 1, min(detection.y2, frame_height))

            # Safe cropping with validated coordinates
            vehicle_crop = frame[y1:y2, x1:x2]

            if vehicle_crop.size == 0:
                logger.error(f"Empty crop for {candidate.journey_id}")
                return None

            # Generate filename with comprehensive journey metadata
            timestamp_str = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(candidate.entry_timestamp)
            )
            duration = exit_event.journey.journey_duration_seconds

            filename = (
                f"{timestamp_str}_{candidate.vehicle_type.value}_"
                f"{candidate.best_confidence:.3f}_{candidate.journey_id}_"
                f"{duration:.1f}s.jpg"
            )

            file_path = self.output_dir / filename
            success = cv2.imwrite(str(file_path), vehicle_crop)

            return file_path if success else None

        except Exception as e:
            logger.error(f"Error saving candidate for {candidate.journey_id}: {e}")
            return None

    def get_statistics(self) -> dict[str, Any]:
        """Get candidate saver statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "saved_candidates": self.saved_count,
            "active_journeys": len(self.active_journeys),
            "output_directory": str(self.output_dir),
            "strategy": "one_best_per_journey",
            "storage_limit_gb": self.storage_limit_bytes / (1024**3),
        }
