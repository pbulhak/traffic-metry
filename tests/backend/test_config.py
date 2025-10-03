"""Tests for configuration management system.

This module contains comprehensive tests for the TrafficMetry configuration
system, including singleton behavior, validation, and test isolation.
"""

import os
import tempfile

import pytest

from backend.config import CameraSettings, Settings


class TestConfigSingleton:
    """Test singleton behavior and reset functionality."""

    def test_get_config_returns_same_instance(self) -> None:
        """Test that get_config returns the same instance on multiple calls."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2, "get_config should return the same instance"

    def test_reset_config_clears_singleton(self) -> None:
        """Test that reset_config clears the singleton instance."""
        # Get initial config
        config1 = get_config()

        # Reset and get new config
        reset_config()
        config2 = get_config()

        assert config1 is not config2, "reset_config should clear singleton"

    def test_config_isolation_between_tests(self) -> None:
        """Test that configuration is isolated between tests."""
        # This test verifies that the conftest.py fixture works
        config = get_config()

        # Modify configuration in memory
        config.debug = True

        # The next test should get a fresh config
        # (verified by fixture auto-reset)


class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_default_configuration_values(self) -> None:
        """Test that default configuration values are set correctly."""
        config = get_config()

        # Test camera defaults
        assert config.camera.url == "rtsp://192.168.1.100:554/stream"
        assert config.camera.width == 1920
        assert config.camera.height == 1080
        assert config.camera.fps == 30

        # Test model defaults
        assert config.model.path == "yolov8n.pt"
        assert config.model.confidence_threshold == 0.5
        assert config.model.device == "cpu"
        assert config.model.max_detections == 100

        # Test server defaults
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000
        assert config.server.cors_origins == ["*"]
        assert config.server.websocket_max_connections == 10

    def test_configuration_with_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration loading with environment variable overrides."""
        # Set environment variables
        monkeypatch.setenv("CAMERA__URL", "rtsp://test.camera:554/stream")
        monkeypatch.setenv("SERVER__PORT", "9000")
        monkeypatch.setenv("DEBUG", "true")

        # Reset to force reload with new env vars
        reset_config()
        config = get_config()

        assert config.camera.url == "rtsp://test.camera:554/stream"
        assert config.server.port == 9000
        assert config.debug is True


class TestLaneConfiguration:
    """Test lane configuration loading and validation."""

    def test_load_lane_config_missing_file(self) -> None:
        """Test lane config loading when file doesn't exist."""
        result = load_lane_config("nonexistent.ini")
        assert result is None

    def test_load_lane_config_valid_file(self) -> None:
        """Test lane config loading with valid INI file."""
        # Create temporary INI file
        ini_content = """[lanes]
line_1 = 100,200,300,400
line_2 = 500,600,700,800

[directions]
1 = left
2 = right
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            temp_path = f.name

        try:
            result = load_lane_config(temp_path)

            assert result is not None
            assert len(result.lines) == 2
            assert result.lines[0] == (100, 200, 300, 400)
            assert result.lines[1] == (500, 600, 700, 800)
            assert result.directions == {1: "left", 2: "right"}
        finally:
            os.unlink(temp_path)

    def test_load_lane_config_invalid_direction(self) -> None:
        """Test lane config validation with invalid direction."""
        # Create temporary INI file with invalid direction
        ini_content = """[directions]
1 = invalid_direction
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            temp_path = f.name

        try:
            with pytest.raises(CalibrationDataMissingError):
                result = load_lane_config(temp_path)
                # Force validation by creating LaneConfig
                if result:
                    LaneConfig(lines=result.lines, directions=result.directions)
        finally:
            os.unlink(temp_path)

    def test_lane_config_direction_validation(self) -> None:
        """Test LaneConfig direction validation."""
        # Valid directions should pass
        valid_config = LaneConfig(
            lines=[(100, 200, 300, 400)], directions={1: "left", 2: "right", 3: "stationary"}
        )
        assert valid_config.directions == {1: "left", 2: "right", 3: "stationary"}

        # Invalid direction should raise ValueError
        with pytest.raises(ValueError, match="Invalid direction"):
            LaneConfig(lines=[], directions={1: "invalid"})


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_camera_settings_validation(self) -> None:
        """Test camera settings field validation."""
        # Valid settings should pass
        valid_camera = CameraSettings(
            url="rtsp://test.com:554/stream", width=1920, height=1080, fps=30
        )
        assert valid_camera.width == 1920

        # Invalid width should raise ValidationError
        with pytest.raises(ValueError):
            CameraSettings(width=500)  # Below minimum of 640

        with pytest.raises(ValueError):
            CameraSettings(width=5000)  # Above maximum of 4096

    def test_model_settings_validation(self) -> None:
        """Test model settings field validation."""
        # Invalid confidence should raise ValidationError
        with pytest.raises(ValueError):
            from backend.config import ModelSettings

            ModelSettings(confidence_threshold=0.05)  # Below minimum of 0.1

        with pytest.raises(ValueError):
            from backend.config import ModelSettings

            ModelSettings(confidence_threshold=1.5)  # Above maximum of 1.0


class TestConfigurationIntegration:
    """Integration tests for complete configuration system."""

    def test_complete_configuration_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test complete configuration loading flow."""
        # Set up environment
        monkeypatch.setenv("CAMERA__URL", "rtsp://integration.test:554/stream")
        monkeypatch.setenv("MODEL__DEVICE", "cuda")

        # Create temporary lane config
        ini_content = """[lanes]
line_1 = 50,100,150,200

[directions]
1 = right
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            temp_path = f.name

        try:
            # Set config file path and reset
            monkeypatch.setenv("CONFIG_FILE", temp_path)
            reset_config()

            # Load configuration
            config = get_config()

            # Verify environment overrides
            assert config.camera.url == "rtsp://integration.test:554/stream"
            assert config.model.device == "cuda"

            # Verify lane configuration loading
            assert config.lanes is not None
            assert len(config.lanes.lines) == 1
            assert config.lanes.lines[0] == (50, 100, 150, 200)
            assert config.lanes.directions == {1: "right"}

        finally:
            os.unlink(temp_path)


class TestROISettings:
    """Test Region of Interest configuration."""

    def test_roi_settings_defaults(self) -> None:
        """Test ROISettings default values."""
        from backend.config import ROISettings

        roi = ROISettings()
        assert roi.enabled is False
        assert roi.x1 == 0
        assert roi.y1 == 0
        assert roi.x2 == 1920
        assert roi.y2 == 1080

    def test_roi_validate_coordinates_disabled(self) -> None:
        """Test validate_coordinates returns True when ROI is disabled."""
        from backend.config import ROISettings

        roi = ROISettings(enabled=False)
        assert roi.validate_coordinates(1920, 1080) is True

    def test_roi_validate_coordinates_valid(self) -> None:
        """Test validate_coordinates with valid ROI."""
        from backend.config import ROISettings

        roi = ROISettings(enabled=True, x1=100, y1=100, x2=1800, y2=980)
        assert roi.validate_coordinates(1920, 1080) is True

    def test_roi_validate_coordinates_invalid_x_order(self) -> None:
        """Test validate_coordinates fails when x1 >= x2."""
        from backend.config import ROISettings

        roi = ROISettings(enabled=True, x1=1000, y1=100, x2=500, y2=980)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_validate_coordinates_invalid_y_order(self) -> None:
        """Test validate_coordinates fails when y1 >= y2."""
        from backend.config import ROISettings

        roi = ROISettings(enabled=True, x1=100, y1=900, x2=1800, y2=200)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_validate_coordinates_exceeds_frame_width(self) -> None:
        """Test validate_coordinates fails when x2 exceeds frame width."""
        from backend.config import ROISettings

        roi = ROISettings(enabled=True, x1=100, y1=100, x2=2000, y2=980)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_validate_coordinates_exceeds_frame_height(self) -> None:
        """Test validate_coordinates fails when y2 exceeds frame height."""
        from backend.config import ROISettings

        roi = ROISettings(enabled=True, x1=100, y1=100, x2=1800, y2=1200)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_get_dimensions(self) -> None:
        """Test get_roi_dimensions returns correct width and height."""
        from backend.config import ROISettings

        roi = ROISettings(enabled=True, x1=400, y1=200, x2=1520, y2=880)
        width, height = roi.get_roi_dimensions()
        assert width == 1120  # 1520 - 400
        assert height == 680  # 880 - 200
