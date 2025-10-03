"""Tests for ROI configuration."""

import pytest

from backend.config import ROISettings


class TestROISettings:
    """Test Region of Interest configuration."""

    def test_roi_settings_defaults(self) -> None:
        """Test ROISettings default values."""
        roi = ROISettings()
        assert roi.enabled is False
        assert roi.x1 == 0
        assert roi.y1 == 0
        assert roi.x2 == 1920
        assert roi.y2 == 1080

    def test_roi_validate_coordinates_disabled(self) -> None:
        """Test validate_coordinates returns True when ROI is disabled."""
        roi = ROISettings(enabled=False)
        assert roi.validate_coordinates(1920, 1080) is True

    def test_roi_validate_coordinates_valid(self) -> None:
        """Test validate_coordinates with valid ROI."""
        roi = ROISettings(enabled=True, x1=100, y1=100, x2=1800, y2=980)
        assert roi.validate_coordinates(1920, 1080) is True

    def test_roi_validate_coordinates_invalid_x_order(self) -> None:
        """Test validate_coordinates fails when x1 >= x2."""
        roi = ROISettings(enabled=True, x1=1000, y1=100, x2=500, y2=980)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_validate_coordinates_invalid_y_order(self) -> None:
        """Test validate_coordinates fails when y1 >= y2."""
        roi = ROISettings(enabled=True, x1=100, y1=900, x2=1800, y2=200)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_validate_coordinates_exceeds_frame_width(self) -> None:
        """Test validate_coordinates fails when x2 exceeds frame width."""
        roi = ROISettings(enabled=True, x1=100, y1=100, x2=2000, y2=980)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_validate_coordinates_exceeds_frame_height(self) -> None:
        """Test validate_coordinates fails when y2 exceeds frame height."""
        roi = ROISettings(enabled=True, x1=100, y1=100, x2=1800, y2=1200)
        assert roi.validate_coordinates(1920, 1080) is False

    def test_roi_get_dimensions(self) -> None:
        """Test get_roi_dimensions returns correct width and height."""
        roi = ROISettings(enabled=True, x1=400, y1=200, x2=1520, y2=880)
        width, height = roi.get_roi_dimensions()
        assert width == 1120  # 1520 - 400
        assert height == 680  # 880 - 200
