"""Unit tests for detection models.

This module contains comprehensive tests for the DetectionResult dataclass
and VehicleType enum, including validation, computed properties, and
factory methods used in the TrafficMetry detection pipeline.
"""

import uuid
from typing import Any, Dict, TYPE_CHECKING

import pytest

from backend.detection_models import (
    COCO_TO_VEHICLE_TYPE,
    DetectionResult,
    VehicleType,
    map_coco_class_to_vehicle_type,
)

if TYPE_CHECKING:
    pass


class TestVehicleType:
    """Test suite for VehicleType enum."""

    def test_vehicle_type_values(self) -> None:
        """Test that VehicleType has correct string values."""
        assert VehicleType.CAR.value == "car"
        assert VehicleType.TRUCK.value == "truck"
        assert VehicleType.BUS.value == "bus"
        assert VehicleType.MOTORCYCLE.value == "motorcycle"
        assert VehicleType.BICYCLE.value == "bicycle"
        assert VehicleType.OTHER_VEHICLE.value == "other_vehicle"

    def test_vehicle_type_is_string_enum(self) -> None:
        """Test that VehicleType values are strings."""
        for vehicle_type in VehicleType:
            assert isinstance(vehicle_type, str)
            assert isinstance(vehicle_type.value, str)

    def test_coco_mapping_constants(self) -> None:
        """Test COCO class ID mapping dictionary."""
        expected_mapping = {
            1: VehicleType.BICYCLE,
            2: VehicleType.CAR,
            3: VehicleType.MOTORCYCLE,
            5: VehicleType.BUS,
            7: VehicleType.TRUCK,
        }
        assert expected_mapping == COCO_TO_VEHICLE_TYPE

    def test_map_coco_class_to_vehicle_type_valid_ids(self) -> None:
        """Test mapping valid COCO class IDs to vehicle types."""
        assert map_coco_class_to_vehicle_type(1) == VehicleType.BICYCLE
        assert map_coco_class_to_vehicle_type(2) == VehicleType.CAR
        assert map_coco_class_to_vehicle_type(3) == VehicleType.MOTORCYCLE
        assert map_coco_class_to_vehicle_type(5) == VehicleType.BUS
        assert map_coco_class_to_vehicle_type(7) == VehicleType.TRUCK

    def test_map_coco_class_to_vehicle_type_invalid_ids(self) -> None:
        """Test mapping invalid COCO class IDs returns OTHER_VEHICLE."""
        assert map_coco_class_to_vehicle_type(0) == VehicleType.OTHER_VEHICLE
        assert map_coco_class_to_vehicle_type(4) == VehicleType.OTHER_VEHICLE
        assert map_coco_class_to_vehicle_type(99) == VehicleType.OTHER_VEHICLE
        assert map_coco_class_to_vehicle_type(-1) == VehicleType.OTHER_VEHICLE


class TestDetectionResult:
    """Test suite for DetectionResult dataclass."""

    @pytest.fixture
    def valid_detection_data(self) -> Dict[str, Any]:
        """Provide valid detection result data for testing."""
        return {
            "x1": 100,
            "y1": 200,
            "x2": 300,
            "y2": 400,
            "confidence": 0.85,
            "class_id": 2,
            "vehicle_type": VehicleType.CAR,
            "detection_id": str(uuid.uuid4()),
            "frame_timestamp": 1672531200.0,
            "frame_id": 42,
            "frame_shape": (1080, 1920, 3),
        }

    def test_detection_result_creation_valid(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test creating DetectionResult with valid data."""
        detection = DetectionResult(**valid_detection_data)

        assert detection.x1 == 100
        assert detection.y1 == 200
        assert detection.x2 == 300
        assert detection.y2 == 400
        assert detection.confidence == 0.85
        assert detection.class_id == 2
        assert detection.vehicle_type == VehicleType.CAR
        assert detection.frame_timestamp == 1672531200.0
        assert detection.frame_id == 42
        assert detection.frame_shape == (1080, 1920, 3)

    def test_detection_result_immutable(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test that DetectionResult is immutable (frozen dataclass)."""
        detection = DetectionResult(**valid_detection_data)

        with pytest.raises(AttributeError):
            detection.x1 = 150  # type: ignore[misc]

        with pytest.raises(AttributeError):
            detection.confidence = 0.9  # type: ignore[misc]

    def test_validation_invalid_bbox_x_coordinates(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test validation fails when x1 >= x2."""
        # x1 == x2
        valid_detection_data["x1"] = 300
        valid_detection_data["x2"] = 300

        with pytest.raises(ValueError, match="x1.*must be less than x2"):
            DetectionResult(**valid_detection_data)

        # x1 > x2
        valid_detection_data["x1"] = 400
        valid_detection_data["x2"] = 300

        with pytest.raises(ValueError, match="x1.*must be less than x2"):
            DetectionResult(**valid_detection_data)

    def test_validation_invalid_bbox_y_coordinates(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test validation fails when y1 >= y2."""
        # y1 == y2
        valid_detection_data["y1"] = 400
        valid_detection_data["y2"] = 400

        with pytest.raises(ValueError, match="y1.*must be less than y2"):
            DetectionResult(**valid_detection_data)

        # y1 > y2
        valid_detection_data["y1"] = 500
        valid_detection_data["y2"] = 400

        with pytest.raises(ValueError, match="y1.*must be less than y2"):
            DetectionResult(**valid_detection_data)

    def test_validation_invalid_confidence_range(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test validation fails for confidence outside 0.0-1.0 range."""
        # Confidence too low
        valid_detection_data["confidence"] = -0.1

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            DetectionResult(**valid_detection_data)

        # Confidence too high
        valid_detection_data["confidence"] = 1.1

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            DetectionResult(**valid_detection_data)

    def test_validation_negative_coordinates(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test validation fails for negative bounding box coordinates."""
        # Negative x1
        valid_detection_data["x1"] = -10

        with pytest.raises(ValueError, match="coordinates cannot be negative"):
            DetectionResult(**valid_detection_data)

        # Reset x1, test negative y1
        valid_detection_data["x1"] = 100
        valid_detection_data["y1"] = -5

        with pytest.raises(ValueError, match="coordinates cannot be negative"):
            DetectionResult(**valid_detection_data)

    def test_validation_bbox_exceeds_frame_boundaries(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test validation fails when bbox extends beyond frame boundaries."""
        frame_height, frame_width, _ = valid_detection_data["frame_shape"]

        # x2 exceeds frame width
        valid_detection_data["x2"] = frame_width + 10

        with pytest.raises(ValueError, match="extends beyond frame boundaries"):
            DetectionResult(**valid_detection_data)

        # Reset x2, test y2 exceeds frame height
        valid_detection_data["x2"] = 300
        valid_detection_data["y2"] = frame_height + 10

        with pytest.raises(ValueError, match="extends beyond frame boundaries"):
            DetectionResult(**valid_detection_data)

    def test_validation_edge_case_boundaries(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test validation for edge cases at frame boundaries."""
        frame_height, frame_width, _ = valid_detection_data["frame_shape"]

        # Valid: bbox exactly at frame boundaries
        valid_detection_data["x2"] = frame_width
        valid_detection_data["y2"] = frame_height

        # Should not raise exception
        detection = DetectionResult(**valid_detection_data)
        assert detection.x2 == frame_width
        assert detection.y2 == frame_height

    def test_centroid_property(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test centroid calculation property."""
        detection = DetectionResult(**valid_detection_data)

        expected_x = (100 + 300) // 2  # 200
        expected_y = (200 + 400) // 2  # 300

        assert detection.centroid == (expected_x, expected_y)
        assert isinstance(detection.centroid[0], int)
        assert isinstance(detection.centroid[1], int)

    def test_bbox_area_pixels_property(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test bounding box area calculation property."""
        detection = DetectionResult(**valid_detection_data)

        expected_area = (300 - 100) * (400 - 200)  # 200 * 200 = 40000

        assert detection.bbox_area_pixels == expected_area
        assert isinstance(detection.bbox_area_pixels, int)

    def test_bbox_width_property(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test bounding box width calculation property."""
        detection = DetectionResult(**valid_detection_data)

        expected_width = 300 - 100  # 200

        assert detection.bbox_width == expected_width
        assert isinstance(detection.bbox_width, int)

    def test_bbox_height_property(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test bounding box height calculation property."""
        detection = DetectionResult(**valid_detection_data)

        expected_height = 400 - 200  # 200

        assert detection.bbox_height == expected_height
        assert isinstance(detection.bbox_height, int)

    def test_aspect_ratio_property(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test aspect ratio calculation property."""
        detection = DetectionResult(**valid_detection_data)

        expected_ratio = 200.0 / 200.0  # 1.0 (square)

        assert detection.aspect_ratio == expected_ratio
        assert isinstance(detection.aspect_ratio, float)

    def test_aspect_ratio_wide_bbox(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test aspect ratio for wide bounding box."""
        valid_detection_data["x2"] = 500  # Width: 400, Height: 200

        detection = DetectionResult(**valid_detection_data)

        expected_ratio = 400.0 / 200.0  # 2.0 (wider than tall)

        assert detection.aspect_ratio == expected_ratio

    def test_aspect_ratio_tall_bbox(self, valid_detection_data: Dict[str, Any]) -> None:
        """Test aspect ratio for tall bounding box."""
        valid_detection_data["y2"] = 600  # Width: 200, Height: 400

        detection = DetectionResult(**valid_detection_data)

        expected_ratio = 200.0 / 400.0  # 0.5 (taller than wide)

        assert detection.aspect_ratio == expected_ratio

    def test_from_yolo_detection_with_provided_id(self) -> None:
        """Test factory method with provided detection ID."""
        detection_id = str(uuid.uuid4())

        detection = DetectionResult.from_yolo_detection(
            x1=50,
            y1=100,
            x2=150,
            y2=250,
            confidence=0.9,
            class_id=2,
            vehicle_type=VehicleType.CAR,
            frame_timestamp=1672531200.0,
            frame_id=10,
            frame_shape=(720, 1280, 3),
            detection_id=detection_id,
        )

        assert detection.detection_id == detection_id
        assert detection.x1 == 50
        assert detection.y1 == 100
        assert detection.confidence == 0.9
        assert detection.vehicle_type == VehicleType.CAR

    def test_from_yolo_detection_auto_generate_id(self) -> None:
        """Test factory method with auto-generated UUID."""
        detection = DetectionResult.from_yolo_detection(
            x1=50,
            y1=100,
            x2=150,
            y2=250,
            confidence=0.9,
            class_id=2,
            vehicle_type=VehicleType.CAR,
            frame_timestamp=1672531200.0,
            frame_id=10,
            frame_shape=(720, 1280, 3),
            detection_id=None,  # Should auto-generate
        )

        # Verify UUID format
        assert isinstance(detection.detection_id, str)
        assert len(detection.detection_id) == 36  # Standard UUID length
        assert detection.detection_id.count("-") == 4  # UUID has 4 hyphens

        # Verify UUID can be parsed
        parsed_uuid = uuid.UUID(detection.detection_id)
        assert str(parsed_uuid) == detection.detection_id

    def test_from_yolo_detection_validation_applies(self) -> None:
        """Test that factory method still applies validation."""
        with pytest.raises(ValueError, match="x1.*must be less than x2"):
            DetectionResult.from_yolo_detection(
                x1=200,  # x1 > x2, should fail validation
                y1=100,
                x2=150,
                y2=250,
                confidence=0.9,
                class_id=2,
                vehicle_type=VehicleType.CAR,
                frame_timestamp=1672531200.0,
                frame_id=10,
                frame_shape=(720, 1280, 3),
            )
