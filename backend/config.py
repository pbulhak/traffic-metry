"""Configuration management for TrafficMetry application.

This module provides centralized configuration management using Pydantic Settings.
Supports loading from environment variables (.env) and INI files (config.ini).
"""

import configparser
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ConfigurationError(Exception):
    """Raised when there's an issue with application configuration."""

    pass


class CalibrationDataMissingError(ConfigurationError):
    """Raised when lane calibration data is missing or invalid."""

    pass


class CameraSettings(BaseModel):
    """Camera-related configuration settings."""

    url: str = Field(
        default="rtsp://192.168.1.100:554/stream", description="RTSP URL for IP camera"
    )
    width: int = Field(default=1920, ge=640, le=4096, description="Camera frame width in pixels")
    height: int = Field(default=1080, ge=480, le=2160, description="Camera frame height in pixels")
    fps: int = Field(default=30, ge=1, le=60, description="Target frames per second")


class ModelSettings(BaseModel):
    """AI model configuration settings."""

    path: str = Field(default="yolov8n.pt", description="Path to YOLO model file")
    confidence_threshold: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Minimum confidence for detections"
    )
    device: str = Field(default="cpu", description="Device for model inference (cpu/cuda)")
    max_detections: int = Field(
        default=100, ge=1, le=1000, description="Maximum detections per frame"
    )


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    path: str = Field(default="data/trafficmetry.db", description="Path to SQLite database file")
    max_records: int = Field(default=1000000, ge=1000, description="Maximum records before cleanup")


class ServerSettings(BaseModel):
    """Web server configuration settings."""

    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    websocket_max_connections: int = Field(
        default=10, ge=1, le=100, description="Max WebSocket connections"
    )


class LaneConfig(BaseModel):
    """Lane configuration loaded from config.ini."""

    lines: list[tuple[int, int, int, int]] = Field(
        default_factory=list, description="Lane divider lines (x1,y1,x2,y2)"
    )
    directions: dict[int, str] = Field(default_factory=dict, description="Lane directions mapping")

    @field_validator("directions")
    @classmethod
    def validate_directions(cls, v: dict[int, str]) -> dict[int, str]:
        """Validate lane direction values."""
        valid_directions = {"left", "right", "stationary"}
        for lane_id, direction in v.items():
            if direction not in valid_directions:
                raise ValueError(f"Invalid direction '{direction}' for lane {lane_id}")
        return v


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
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Lane configuration (loaded separately)
    lanes: LaneConfig | None = Field(default=None)

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    config_file: str = Field(default="config.ini", description="Path to calibration config file")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False


def load_lane_config(config_path: str) -> LaneConfig | None:
    """Load lane configuration from INI file.

    Args:
        config_path: Path to the config.ini file

    Returns:
        LaneConfig instance or None if file doesn't exist

    Raises:
        CalibrationDataMissingError: If config file exists but is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return None

    try:
        parser = configparser.ConfigParser()
        parser.read(config_path)

        # Parse lane lines
        lines = []
        if parser.has_section("lanes"):
            for key, value in parser["lanes"].items():
                if key.startswith("line_"):
                    coords = [int(x.strip()) for x in value.split(",")]
                    if len(coords) == 4:
                        lines.append((coords[0], coords[1], coords[2], coords[3]))

        # Parse lane directions
        directions = {}
        if parser.has_section("directions"):
            for lane_id, direction in parser["directions"].items():
                directions[int(lane_id)] = direction.strip()

        return LaneConfig(lines=lines, directions=directions)

    except (configparser.Error, ValueError) as e:
        raise CalibrationDataMissingError(f"Invalid calibration data in {config_path}: {e}") from e


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

        # Load lane configuration if available
        settings.lanes = load_lane_config(settings.config_file)

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