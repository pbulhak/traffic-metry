"""Data models for vehicle detection results.

This module defines the core data structures used for representing
vehicle detections from YOLO models. These are internal representations
optimized for processing within the TrafficMetry system.

The DetectionResult class represents a single vehicle detection with
bounding box coordinates in pixel space, confidence scores, and metadata
for tracking and analysis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum


class VehicleType(str, Enum):
    """Enumeration of supported vehicle types.

    Maps COCO dataset vehicle classes to TrafficMetry vehicle categories.
    Values are compatible with API v2.3 contract specifications.

    COCO class mappings:
    - CAR: COCO class 2 (car)
    - MOTORCYCLE: COCO class 3 (motorcycle)
    - BUS: COCO class 5 (bus)
    - TRUCK: COCO class 7 (truck)
    - BICYCLE: COCO class 1 (bicycle) - for future use
    """

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    OTHER_VEHICLE = "other_vehicle"


@dataclass(frozen=True)
class DetectionResult:
    """Represents a single vehicle detection result.

    This is the internal representation of a vehicle detection from YOLO,
    containing bounding box coordinates in pixel space, confidence metrics,
    and metadata for tracking and further processing.

    Bounding box coordinates are in absolute pixel values relative to the
    source frame, following the format: (x1, y1) = top-left corner,
    (x2, y2) = bottom-right corner.

    Attributes:
        x1: Left edge of bounding box in pixels
        y1: Top edge of bounding box in pixels
        x2: Right edge of bounding box in pixels
        y2: Bottom edge of bounding box in pixels
        confidence: Detection confidence score (0.0-1.0)
        class_id: COCO dataset class identifier
        vehicle_type: Classified vehicle type
        detection_id: Unique identifier for this detection
        frame_timestamp: Unix timestamp when frame was captured
        frame_id: Sequential frame number for ordering
        frame_shape: Original frame dimensions (height, width, channels)
        track_id: Consistent tracking ID assigned by ObjectTracker (None for raw detections)
    """

    # Bounding box coordinates in pixels (integer values)
    x1: int
    y1: int
    x2: int
    y2: int

    # Detection metadata
    confidence: float
    class_id: int
    vehicle_type: VehicleType

    # Tracking and temporal information
    detection_id: str
    frame_timestamp: float
    frame_id: int
    frame_shape: tuple
    track_id: int | None = None  # Assigned by ObjectTracker, None for raw detections

    def __post_init__(self) -> None:
        """Validate detection result data after initialization."""
        if self.x1 >= self.x2:
            raise ValueError(
                f"Invalid bounding box: x1 ({self.x1}) must be less than x2 ({self.x2})"
            )

        if self.y1 >= self.y2:
            raise ValueError(
                f"Invalid bounding box: y1 ({self.y1}) must be less than y2 ({self.y2})"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("Bounding box coordinates cannot be negative")

        frame_height, frame_width, _ = self.frame_shape
        if self.x2 > frame_width or self.y2 > frame_height:
            raise ValueError("Bounding box extends beyond frame boundaries")

    @property
    def centroid(self) -> tuple[int, int]:
        """Calculate the center point of the bounding box.

        Returns:
            Tuple of (x, y) coordinates of the bounding box center in pixels.
        """
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return (center_x, center_y)

    @property
    def bbox_area_pixels(self) -> int:
        """Calculate the area of the bounding box in pixels.

        Returns:
            Area of the bounding box in square pixels.
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def bbox_width(self) -> int:
        """Get the width of the bounding box.

        Returns:
            Width of the bounding box in pixels.
        """
        return self.x2 - self.x1

    @property
    def bbox_height(self) -> int:
        """Get the height of the bounding box.

        Returns:
            Height of the bounding box in pixels.
        """
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio of the bounding box.

        Returns:
            Aspect ratio as width/height. Values > 1.0 indicate
            wider than tall, values < 1.0 indicate taller than wide.
        """
        return self.bbox_width / self.bbox_height

    @classmethod
    def from_yolo_detection(
        cls,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
        class_id: int,
        vehicle_type: VehicleType,
        frame_timestamp: float,
        frame_id: int,
        frame_shape: tuple,
        detection_id: str | None = None,
        track_id: int | None = None,
    ) -> DetectionResult:
        """Create DetectionResult from YOLO model output.

        Factory method for creating DetectionResult instances from raw
        YOLO detection data with automatic ID generation if not provided.

        Args:
            x1: Left edge of bounding box in pixels
            y1: Top edge of bounding box in pixels
            x2: Right edge of bounding box in pixels
            y2: Bottom edge of bounding box in pixels
            confidence: Detection confidence score (0.0-1.0)
            class_id: COCO dataset class identifier
            vehicle_type: Classified vehicle type
            frame_timestamp: Unix timestamp when frame was captured
            frame_id: Sequential frame number
            frame_shape: Original frame dimensions (height, width, channels)
            detection_id: Optional unique identifier, generated if None
            track_id: Optional tracking ID from ObjectTracker

        Returns:
            DetectionResult instance ready for further processing.
        """
        if detection_id is None:
            detection_id = str(uuid.uuid4())

        return cls(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=confidence,
            class_id=class_id,
            vehicle_type=vehicle_type,
            detection_id=detection_id,
            frame_timestamp=frame_timestamp,
            frame_id=frame_id,
            frame_shape=frame_shape,
            track_id=track_id,
        )


# COCO class ID to VehicleType mapping for YOLO integration
COCO_TO_VEHICLE_TYPE = {
    1: VehicleType.BICYCLE,
    2: VehicleType.CAR,
    3: VehicleType.MOTORCYCLE,
    5: VehicleType.BUS,
    7: VehicleType.TRUCK,
}


def map_coco_class_to_vehicle_type(class_id: int) -> VehicleType:
    """Map COCO dataset class ID to VehicleType enum.

    Args:
        class_id: COCO dataset class identifier from YOLO detection

    Returns:
        Corresponding VehicleType enum value, or OTHER_VEHICLE for
        unrecognized class IDs.
    """
    return COCO_TO_VEHICLE_TYPE.get(class_id, VehicleType.OTHER_VEHICLE)
