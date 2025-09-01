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
    """Essential vehicle journey data for database storage.

    Simple representation of a vehicle journey from entry to exit
    for MVP requirements without YAGNI analytics.
    """
    track_id: int
    vehicle_type: VehicleType
    entry_timestamp: float
    exit_timestamp: float | None
    entry_lane: int | None
    exit_lane: int | None
    movement_direction: str | None
    total_detections: int
    best_confidence: float
    best_bbox: tuple[int, int, int, int]
    best_detection_timestamp: float
    journey_duration_seconds: float

    def to_api_format(self) -> dict[str, Any]:
        """Convert journey to API v2.3 compatible format for final reporting."""
        return {
            "eventId": f"journey_{self.track_id}",
            "timestamp": self.exit_timestamp or self.entry_timestamp,
            "vehicleId": f"vehicle_{self.track_id}",
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": self.movement_direction,
                "lane": self.exit_lane or self.entry_lane
            },
            "position": {
                "boundingBox": {
                    "x1": self.best_bbox[0],
                    "y1": self.best_bbox[1],
                    "x2": self.best_bbox[2],
                    "y2": self.best_bbox[3]
                }
            },
            "analytics": {
                "confidence": self.best_confidence,
                "journeyDurationSeconds": self.journey_duration_seconds,
                "totalDetections": self.total_detections
            }
        }


@dataclass(frozen=True)
class VehicleEvent(ABC):
    """Base class for all vehicle lifecycle events."""
    track_id: int
    timestamp: float
    vehicle_type: VehicleType

    @abstractmethod
    def to_websocket_format(self) -> dict[str, Any]:
        """Convert event to WebSocket message format."""
        pass


@dataclass(frozen=True)
class VehicleEntered(VehicleEvent):
    """Event: New vehicle entered the tracking area.
    
    Triggered when ByteTrack creates a new track ID for a vehicle.
    Used for logging and initial system notifications.
    """
    detection: DetectionResult
    lane: int | None
    direction: str | None

    def to_websocket_format(self) -> dict[str, Any]:
        """Convert to WebSocket format for real-time notifications."""
        return {
            "type": "vehicle_entered",
            "eventId": f"enter_{self.track_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": f"vehicle_{self.track_id}",
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": self.direction,
                "lane": self.lane
            },
            "position": {
                "boundingBox": {
                    "x1": self.detection.x1,
                    "y1": self.detection.y1,
                    "x2": self.detection.x2,
                    "y2": self.detection.y2
                }
            },
            "analytics": {
                "confidence": self.detection.confidence,
                "isNewVehicle": True
            }
        }


@dataclass(frozen=True)
class VehicleUpdated(VehicleEvent):
    """Event: Vehicle position/state updated - for real-time WebSocket.
    
    Triggered periodically for active vehicles to provide real-time
    position updates to connected frontend clients.
    """
    detection: DetectionResult
    lane: int | None
    direction: str | None
    total_detections_so_far: int
    current_confidence: float

    def to_websocket_format(self) -> dict[str, Any]:
        """Convert to API v2.3 compatible WebSocket event."""
        return {
            "eventId": f"update_{self.track_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": f"vehicle_{self.track_id}",
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": self.direction,
                "lane": self.lane
            },
            "vehicleColor": {
                "hex": None,  # Not implemented yet
                "name": None
            },
            "position": {
                "boundingBox": {
                    "x1": self.detection.x1,
                    "y1": self.detection.y1,
                    "x2": self.detection.x2,
                    "y2": self.detection.y2
                }
            },
            "analytics": {
                "confidence": self.current_confidence,
                "estimatedSpeedKph": None,  # Not implemented yet
                "detectionsCount": self.total_detections_so_far
            }
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
        """Convert to WebSocket format for exit notifications."""
        return {
            "type": "vehicle_exited",
            "eventId": f"exit_{self.track_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": f"vehicle_{self.track_id}",
            "vehicleType": self.vehicle_type.value,
            "exitReason": self.exit_reason,
            "journeySummary": {
                "duration": self.journey.journey_duration_seconds,
                "totalDetections": self.journey.total_detections,
                "entryLane": self.journey.entry_lane,
                "exitLane": self.journey.exit_lane
            }
        }


# Event type union for type safety
VehicleEventType = VehicleEntered | VehicleUpdated | VehicleExited
