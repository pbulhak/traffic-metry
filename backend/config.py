"""Configuration management for TrafficMetry application.

This module provides centralized configuration management using Pydantic Settings.
Supports loading from environment variables (.env).
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ConfigurationError(Exception):
    """Raised when there's an issue with application configuration."""

    pass


class CameraSettings(BaseModel):
    """Camera-related configuration settings."""

    url: str = Field(
        default="rtsp://192.168.1.100:554/stream", description="RTSP URL for IP camera"
    )
    width: int = Field(default=1920, ge=640, le=4096, description="Camera frame width in pixels")
    height: int = Field(default=1080, ge=480, le=2160, description="Camera frame height in pixels")
    fps: int = Field(default=25, ge=1, le=60, description="Camera frame rate for video processing")


class ModelSettings(BaseModel):
    """AI model configuration settings."""

    path: str = Field(
        default="data/models/yolov8n.pt",
        description="Path to YOLO model file (.pt) or OpenVINO folder",
    )
    device: str = Field(default="cpu", description="Device for model inference (cpu/cuda)")

    # Detection confidence thresholds
    confidence_threshold: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Threshold for raw YOLO detections"
    )
    track_activation_threshold: float = Field(
        default=0.3, ge=0.1, le=0.9, description="Threshold for ByteTrack to activate new track"
    )

    # Tracker parameters
    minimum_matching_threshold: float = Field(
        default=0.8, ge=0.3, le=0.99, description="IoU threshold for matching detections to tracks"
    )
    lost_track_buffer: int = Field(
        default=30, ge=1, le=120, description="Frames to remember lost object before completion"
    )
    minimum_consecutive_frames: int = Field(
        default=5, ge=1, le=20, description="Consecutive frames required to confirm track"
    )

    # Candidate saver settings
    candidate_storage_limit_gb: float = Field(
        default=10.0, gt=0, le=1000.0, description="Storage limit for candidate images in GB"
    )


class ROISettings(BaseModel):
    """Region of Interest configuration for detection optimization."""

    enabled: bool = Field(
        default=False, description="Enable ROI-based detection (processes only specific region)"
    )
    x1: int = Field(default=0, ge=0, description="ROI top-left X coordinate (pixels)")
    y1: int = Field(default=0, ge=0, description="ROI top-left Y coordinate (pixels)")
    x2: int = Field(default=1920, ge=0, description="ROI bottom-right X coordinate (pixels)")
    y2: int = Field(default=1080, ge=0, description="ROI bottom-right Y coordinate (pixels)")

    def validate_coordinates(self, frame_width: int, frame_height: int) -> bool:
        """Validate ROI coordinates against frame dimensions.

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            True if valid, False otherwise
        """
        if not self.enabled:
            return True

        if self.x1 >= self.x2 or self.y1 >= self.y2:
            return False

        if self.x2 > frame_width or self.y2 > frame_height:
            return False

        return True

    def get_roi_dimensions(self) -> tuple[int, int]:
        """Get ROI dimensions (width, height).

        Returns:
            Tuple of (width, height) in pixels
        """
        return (self.x2 - self.x1, self.y2 - self.y1)


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    path: str = Field(default="data/trafficmetry.db", description="Path to SQLite database file")


class ServerSettings(BaseModel):
    """Web server configuration settings."""

    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    websocket_max_connections: int = Field(
        default=10, ge=1, le=100, description="Max WebSocket connections"
    )

    # Vehicle tracking configuration
    tracker_update_interval_seconds: float = Field(
        default=1.0, gt=0.0, le=10.0, description="Minimum interval between VehicleUpdated events"
    )


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    level: str = Field(default="INFO", description="Log level")
    file_path: str | None = Field(default=None, description="Log file path (None for console)")
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Max log file size in MB")


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    # Sub-settings
    camera: CameraSettings = Field(default_factory=CameraSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    roi: ROISettings = Field(default_factory=ROISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False


def load_config() -> Settings:
    """Factory function to load complete application configuration.

    Returns:
        Configured Settings instance

    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    try:
        # Load main settings from .env
        settings = Settings()

        return settings

    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


# Global configuration instance (lazy-loaded)
_config: Settings | None = None


def get_config() -> Settings:
    """Get the global configuration instance (singleton pattern).

    Returns:
        Global Settings instance
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset global configuration instance (for testing purposes only).

    This function clears the global configuration singleton, forcing
    get_config() to reload configuration on next call. Intended for
    use in test environments to ensure test isolation.
    """
    global _config
    _config = None
