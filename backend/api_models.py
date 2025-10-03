"""API models for TrafficMetry WebSocket communication.

This module provides minimal Pydantic models for WebSocket event communication,
ensuring consistency with the EventGenerator v2.3 contract.
"""

from __future__ import annotations

from datetime import datetime

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
