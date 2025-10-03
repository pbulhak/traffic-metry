"""Vehicle lifecycle events for event-driven TrafficMetry architecture.

This module defines the core event system for tracking vehicle journeys from entry
to exit, enabling efficient database storage and real-time WebSocket updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from backend.detection_models import DetectionResult, VehicleType


@dataclass(frozen=True)
class VehicleJourney:
    """Simplified vehicle journey data with clean position tracking.

    Position-based journey representation without static lane dependencies,
    featuring essential journey data and unique identifiers.
    """

    # Unique identifiers
    journey_id: str  # Global unique ID (JOURNEY_000001) - SINGLE SOURCE OF TRUTH

    # Basic data
    vehicle_type: VehicleType
    entry_timestamp: float
    exit_timestamp: float | None
    journey_duration_seconds: float

    # Clean position data (dedicated columns)
    entry_pos_x: int
    entry_pos_y: int
    exit_pos_x: int
    exit_pos_y: int
    movement_direction: str | None  # Dynamic: "left", "right", "stationary"

    # Essential metrics
    total_detections: int
    best_confidence: float
    best_bbox: tuple[int, int, int, int]
    best_detection_timestamp: float

    def to_api_format(self) -> dict[str, Any]:
        """Convert journey to API v2.3 compatible format with clean data."""
        return {
            "eventId": f"journey_{self.journey_id}",
            "timestamp": self.exit_timestamp or self.entry_timestamp,
            "vehicleId": self.journey_id,
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": self.movement_direction,
                "entryPosition": {"x": self.entry_pos_x, "y": self.entry_pos_y},
                "exitPosition": {"x": self.exit_pos_x, "y": self.exit_pos_y},
            },
            "position": {
                "boundingBox": {
                    "x1": self.best_bbox[0],
                    "y1": self.best_bbox[1],
                    "x2": self.best_bbox[2],
                    "y2": self.best_bbox[3],
                }
            },
            "analytics": {
                "confidence": self.best_confidence,
                "journeyDurationSeconds": self.journey_duration_seconds,
                "totalDetections": self.total_detections,
            },
        }


@dataclass(frozen=True)
class VehicleEvent(ABC):
    """Base class for all vehicle lifecycle events with unique journey identifiers."""

    track_id: int
    journey_id: str
    timestamp: float
    vehicle_type: VehicleType

    @abstractmethod
    def to_websocket_format(self) -> dict[str, Any]:
        """Convert event to WebSocket message format."""
        pass


@dataclass(frozen=True)
class VehicleEntered(VehicleEvent):
    """Event: New vehicle entered the tracking area with unique journey ID.

    Triggered when ByteTrack creates a new track ID for a vehicle.
    Used for logging and initial system notifications with dynamic direction tracking.
    """

    detection: DetectionResult

    def to_websocket_format(self) -> dict[str, Any]:
        """Convert to WebSocket format with clean, standardized movement structure."""
        return {
            "type": "vehicle_entered",
            "eventId": f"enter_{self.journey_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": self.journey_id,
            "trackId": self.track_id,
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": None,  # Direction will be determined dynamically
                "entryPosition": {"x": self.detection.centroid[0], "y": self.detection.centroid[1]},
            },
            "position": {
                "boundingBox": {
                    "x1": self.detection.x1,
                    "y1": self.detection.y1,
                    "x2": self.detection.x2,
                    "y2": self.detection.y2,
                }
            },
            "analytics": {
                "confidence": self.detection.confidence,
                "isNewVehicle": True,
                "journeyStarted": True,
            },
        }


@dataclass(frozen=True)
class VehicleUpdated(VehicleEvent):
    """Event: Vehicle position/state updated with dynamic direction analysis.

    Triggered periodically for active vehicles to provide real-time
    position updates and dynamic direction analysis to connected frontend clients.
    """

    detection: DetectionResult
    movement_direction: str | None
    total_detections_so_far: int
    current_confidence: float

    def to_websocket_format(self) -> dict[str, Any]:
        """Convert to enhanced WebSocket event with dynamic direction data."""
        return {
            "eventId": f"update_{self.journey_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": self.journey_id,
            "trackId": self.track_id,
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": self.movement_direction,
                "currentPosition": {
                    "x": self.detection.centroid[0],
                    "y": self.detection.centroid[1],
                },
            },
            "vehicleColor": {
                "hex": None,  # Not implemented yet
                "name": None,
            },
            "position": {
                "boundingBox": {
                    "x1": self.detection.x1,
                    "y1": self.detection.y1,
                    "x2": self.detection.x2,
                    "y2": self.detection.y2,
                }
            },
            "analytics": {
                "confidence": self.current_confidence,
                "estimatedSpeedKph": None,  # Not implemented yet
                "detectionsCount": self.total_detections_so_far,
            },
        }


@dataclass(frozen=True)
class VehicleExited(VehicleEvent):
    """Event: Vehicle exited tracking area - ready for database storage.

    Triggered when ByteTrack loses a track (vehicle left frame or
    confidence dropped too low). Contains complete journey data.
    """

    journey: VehicleJourney
    exit_reason: str  # "lost_track", "boundary_exit", "low_confidence"

    def to_websocket_format(self) -> dict[str, Any]:
        """Convert to WebSocket format with consistent movement structure.

        Returns event with standardized movement.direction field for frontend consistency.
        """
        return {
            "type": "vehicle_exited",
            "eventId": f"exit_{self.journey_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": self.journey_id,
            "trackId": self.track_id,
            "vehicleType": self.vehicle_type.value,
            "exitReason": self.exit_reason,
            "movement": {
                "direction": self.journey.movement_direction,
                "entryPosition": {"x": self.journey.entry_pos_x, "y": self.journey.entry_pos_y},
                "exitPosition": {"x": self.journey.exit_pos_x, "y": self.journey.exit_pos_y},
            },
            "journeySummary": {
                "duration": self.journey.journey_duration_seconds,
                "totalDetections": self.journey.total_detections,
                "bestConfidence": self.journey.best_confidence,
            },
        }


# Event type union for type safety
VehicleEventType = VehicleEntered | VehicleUpdated | VehicleExited
