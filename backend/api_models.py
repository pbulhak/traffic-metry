"""API response models for TrafficMetry FastAPI endpoints.

This module provides Pydantic models for API responses, ensuring consistency
with the EventGenerator v2.3 contract and providing proper validation for
client-server communication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VehicleMovement(BaseModel):
    """Vehicle movement information."""

    direction: str = Field(..., description="Movement direction: left, right, or stationary")
    lane: int = Field(..., description="Lane number (-1 if unassigned)")


class VehicleColor(BaseModel):
    """Vehicle color information."""

    hex: str | None = Field(None, description="Hex color code if detected")
    name: str | None = Field(None, description="Color name if detected")


class BoundingBox(BaseModel):
    """Vehicle bounding box coordinates."""

    x1: int = Field(..., description="Top-left X coordinate")
    y1: int = Field(..., description="Top-left Y coordinate")
    x2: int = Field(..., description="Bottom-right X coordinate")
    y2: int = Field(..., description="Bottom-right Y coordinate")


class VehiclePosition(BaseModel):
    """Vehicle position information."""

    boundingBox: BoundingBox = Field(..., alias="boundingBox")


class VehicleAnalytics(BaseModel):
    """Vehicle detection analytics."""

    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    estimatedSpeedKph: float | None = Field(None, description="Estimated speed in km/h")


class VehicleEvent(BaseModel):
    """Complete vehicle detection event matching API v2.3 contract.

    This model represents a single vehicle detection event as generated
    by EventGenerator and stored in EventDatabase.
    """

    eventId: str = Field(..., description="Unique event identifier")
    timestamp: str = Field(..., description="Event timestamp in ISO format")
    vehicleId: str = Field(..., description="Vehicle tracking identifier")
    vehicleType: str = Field(
        ..., description="Vehicle type: car, truck, bus, motorcycle, bicycle, or other_vehicle"
    )
    movement: VehicleMovement = Field(..., description="Vehicle movement information")
    vehicleColor: VehicleColor = Field(..., description="Vehicle color information")
    position: VehiclePosition = Field(..., description="Vehicle position information")
    analytics: VehicleAnalytics = Field(..., description="Detection analytics")

    class Config:
        """Pydantic configuration."""

        allow_population_by_field_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class EventsResponse(BaseModel):
    """Response model for /api/events endpoint."""

    events: list[VehicleEvent] = Field(default_factory=list, description="List of vehicle events")
    total_count: int = Field(..., ge=0, description="Total number of events returned")
    timestamp: str = Field(..., description="Response timestamp")


class DatabaseStats(BaseModel):
    """Database statistics information."""

    database_path: str = Field(..., description="Path to database file")
    record_count: int = Field(..., ge=0, description="Total number of records")
    database_size_bytes: int = Field(..., ge=0, description="Database file size in bytes")
    database_size_mb: float = Field(..., ge=0, description="Database file size in MB")
    events_saved: int = Field(..., ge=0, description="Events saved in current session")
    oldest_event: str | None = Field(None, description="Timestamp of oldest event")
    newest_event: str | None = Field(None, description="Timestamp of newest event")
    preservation_mode: str = Field(..., description="Database preservation mode")


class SystemHealth(BaseModel):
    """System health information."""

    status: str = Field(..., description="Overall system status: healthy, degraded, or unhealthy")
    uptime_seconds: float = Field(..., ge=0, description="System uptime in seconds")
    database_connected: bool = Field(..., description="Database connection status")
    camera_connected: bool = Field(..., description="Camera connection status")
    detection_active: bool = Field(..., description="Vehicle detection status")
    websocket_connections: int = Field(..., ge=0, description="Active WebSocket connections")
    last_event_time: str | None = Field(None, description="Timestamp of last detected event")


class StatsResponse(BaseModel):
    """Response model for /api/stats endpoint."""

    database: DatabaseStats = Field(..., description="Database statistics")
    system: SystemHealth = Field(..., description="System health information")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Response model for /api/health endpoint."""

    status: str = Field(..., description="Health check status: ok, warning, or error")
    checks: dict[str, Any] = Field(default_factory=dict, description="Individual health checks")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(default="1.0.0", description="API version")


class WebSocketMessage(BaseModel):
    """WebSocket message wrapper."""

    type: str = Field(..., description="Message type: event, status, or error")
    data: dict[str, Any] = Field(..., description="Message payload")
    timestamp: str = Field(..., description="Message timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
