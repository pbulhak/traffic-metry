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
    """Enhanced vehicle journey data with dynamic direction detection and movement analytics.

    Position-based journey representation without static lane dependencies,
    featuring dynamic movement analysis and unique journey identifiers.
    """
    # Unique identifiers
    track_id: int  # ByteTrack ID (can be recycled)
    journey_id: str  # Global unique ID (JOURNEY_000001)
    
    # Basic data
    vehicle_type: VehicleType
    entry_timestamp: float
    exit_timestamp: float | None
    journey_duration_seconds: float
    
    # Position-based data (replaces lane data)
    entry_position: tuple[int, int]  # Entry centroid (x, y)
    exit_position: tuple[int, int]   # Exit centroid (x, y)
    movement_direction: str | None   # Dynamic: "left", "right", "stationary"
    direction_confidence: float      # 0.0-1.0 confidence in direction
    
    # Movement analytics
    total_movement_pixels: float     # Total distance traveled
    average_speed_pixels_per_second: float
    displacement_vector: tuple[float, float]  # (dx, dy) overall displacement
    
    # Essential metrics
    total_detections: int
    best_confidence: float
    best_bbox: tuple[int, int, int, int]
    best_detection_timestamp: float

    def to_api_format(self) -> dict[str, Any]:
        """Convert journey to API v2.3 compatible format with enhanced analytics."""
        return {
            "eventId": f"journey_{self.journey_id}",
            "timestamp": self.exit_timestamp or self.entry_timestamp,
            "vehicleId": self.journey_id,
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": self.movement_direction,
                "directionConfidence": self.direction_confidence,
                "entryPosition": {"x": self.entry_position[0], "y": self.entry_position[1]},
                "exitPosition": {"x": self.exit_position[0], "y": self.exit_position[1]},
                "displacement": {"dx": self.displacement_vector[0], "dy": self.displacement_vector[1]}
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
                "totalDetections": self.total_detections,
                "totalMovementPixels": self.total_movement_pixels,
                "averageSpeedPixelsPerSecond": self.average_speed_pixels_per_second
            }
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
        """Convert to WebSocket format for real-time notifications."""
        return {
            "type": "vehicle_entered",
            "eventId": f"enter_{self.journey_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": self.journey_id,
            "trackId": self.track_id,
            "vehicleType": self.vehicle_type.value,
            "movement": {
                "direction": None,  # Direction will be determined dynamically
                "directionConfidence": 0.0,
                "entryPosition": {"x": self.detection.centroid[0], "y": self.detection.centroid[1]}
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
                "isNewVehicle": True,
                "journeyStarted": True
            }
        }


@dataclass(frozen=True)
class VehicleUpdated(VehicleEvent):
    """Event: Vehicle position/state updated with dynamic direction analysis.
    
    Triggered periodically for active vehicles to provide real-time
    position updates and dynamic direction analysis to connected frontend clients.
    """
    detection: DetectionResult
    movement_direction: str | None
    direction_confidence: float
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
                "directionConfidence": self.direction_confidence,
                "currentPosition": {"x": self.detection.centroid[0], "y": self.detection.centroid[1]}
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
                "detectionsCount": self.total_detections_so_far,
                "dynamicDirectionAnalysis": True
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
        """Convert to WebSocket format for comprehensive journey completion notifications."""
        return {
            "type": "vehicle_exited",
            "eventId": f"exit_{self.journey_id}_{int(self.timestamp)}",
            "timestamp": self.timestamp,
            "vehicleId": self.journey_id,
            "trackId": self.track_id,
            "vehicleType": self.vehicle_type.value,
            "exitReason": self.exit_reason,
            "journeySummary": {
                "duration": self.journey.journey_duration_seconds,
                "totalDetections": self.journey.total_detections,
                "entryPosition": {"x": self.journey.entry_position[0], "y": self.journey.entry_position[1]},
                "exitPosition": {"x": self.journey.exit_position[0], "y": self.journey.exit_position[1]},
                "movementDirection": self.journey.movement_direction,
                "directionConfidence": self.journey.direction_confidence,
                "totalMovementPixels": self.journey.total_movement_pixels,
                "averageSpeedPixelsPerSecond": self.journey.average_speed_pixels_per_second,
                "displacement": {
                    "dx": self.journey.displacement_vector[0], 
                    "dy": self.journey.displacement_vector[1]
                }
            }
        }


# Event type union for type safety
VehicleEventType = VehicleEntered | VehicleUpdated | VehicleExited
