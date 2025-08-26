"""Comprehensive unit tests for VehicleDetector.

This module provides complete test coverage for the VehicleDetector class,
including initialization, model loading, frame detection, error handling,
and utility functions. Tests use extensive mocking to avoid dependencies
on actual YOLO models and ultralytics imports.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from backend.config import ModelSettings
from backend.detection_models import DetectionResult, VehicleType
from backend.detector import DetectionError, ModelLoadError, VehicleDetector


class TestVehicleDetectorInitialization:
    """Test VehicleDetector initialization and configuration validation."""

    def test_valid_initialization(self, sample_model_settings: ModelSettings) -> None:
        """Test successful detector initialization with valid settings."""
        detector = VehicleDetector(sample_model_settings)

        assert detector.model_settings == sample_model_settings
        assert detector._model is None
        assert detector._model_loaded is False
        assert detector._frame_counter == 0
        assert {1, 2, 3, 5, 7} == detector.VEHICLE_CLASS_IDS

    def test_initialization_with_string_path(self) -> None:
        """Test initialization with string model path."""
        settings = ModelSettings(path="models/yolo11n.pt", device="cpu", confidence_threshold=0.5)

        detector = VehicleDetector(settings)
        assert str(detector.model_settings.path) == "models/yolo11n.pt"

    def test_initialization_with_path_object(self, tmp_path: Path) -> None:
        """Test initialization with Path object."""
        model_path = tmp_path / "model.pt"
        settings = ModelSettings(path=str(model_path), device="cpu", confidence_threshold=0.5)

        detector = VehicleDetector(settings)
        assert str(detector.model_settings.path) == str(model_path)

    def test_invalid_model_path_type(self) -> None:
        """Test initialization fails with invalid model path type."""
        with pytest.raises(ValueError, match="Input should be a valid string"):
            ModelSettings(
                path=123,  # type: ignore  # Invalid type
                device="cpu",
                confidence_threshold=0.5,
            )

    def test_invalid_confidence_threshold_low(self) -> None:
        """Test initialization fails with confidence threshold below 0.1."""
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0.1"):
            ModelSettings(path="model.pt", device="cpu", confidence_threshold=-0.1)

    def test_invalid_confidence_threshold_high(self) -> None:
        """Test initialization fails with confidence threshold above 1."""
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            ModelSettings(path="model.pt", device="cpu", confidence_threshold=1.1)

    def test_detector_validation_logic(self, sample_model_settings: ModelSettings) -> None:
        """Test VehicleDetector's own validation logic."""
        detector = VehicleDetector(sample_model_settings)

        # Test that detector properly stores settings and initializes state
        assert detector.model_settings == sample_model_settings
        assert detector._model is None
        assert detector._model_loaded is False
        assert detector._frame_counter == 0

    def test_repr_string(self, sample_model_settings: ModelSettings) -> None:
        """Test string representation of VehicleDetector."""
        detector = VehicleDetector(sample_model_settings)

        expected = (
            f"VehicleDetector("
            f"model_path='{sample_model_settings.path}', "
            f"device='{sample_model_settings.device}', "
            f"confidence={sample_model_settings.confidence_threshold}, "
            f"loaded=False)"
        )
        assert repr(detector) == expected


class TestVehicleDetectorModelLoading:
    """Test model loading functionality and error handling."""

    @patch("backend.detector.Path.exists")
    @patch("ultralytics.YOLO")
    def test_successful_model_loading(
        self, mock_yolo: Mock, mock_exists: Mock, sample_model_settings: ModelSettings
    ) -> None:
        """Test successful YOLO model loading."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model

        detector = VehicleDetector(sample_model_settings)
        detector._load_model()

        assert detector._model_loaded is True
        assert detector._model == mock_model
        mock_yolo.assert_called_once_with(str(sample_model_settings.path))
        mock_model.to.assert_called_once_with(sample_model_settings.device)

    @patch("backend.detector.Path.exists")
    def test_model_file_not_found(
        self, mock_exists: Mock, sample_model_settings: ModelSettings
    ) -> None:
        """Test ModelLoadError when model file doesn't exist."""
        mock_exists.return_value = False

        detector = VehicleDetector(sample_model_settings)

        with pytest.raises(ModelLoadError) as exc_info:
            detector._load_model()

        assert "Model file not found" in str(exc_info.value)
        assert exc_info.value.model_path == str(sample_model_settings.path)
        assert detector._model_loaded is False

    @patch("backend.detector.Path.exists")
    def test_ultralytics_import_error(
        self, mock_exists: Mock, sample_model_settings: ModelSettings
    ) -> None:
        """Test ModelLoadError when ultralytics import fails."""
        mock_exists.return_value = True

        with patch("builtins.__import__", side_effect=ImportError("ultralytics not found")):
            detector = VehicleDetector(sample_model_settings)

            with pytest.raises(ModelLoadError) as exc_info:
                detector._load_model()

            assert "Failed to import ultralytics YOLO" in str(exc_info.value)
            assert "Install with: pip install ultralytics" in str(exc_info.value)
            assert exc_info.value.model_path == str(sample_model_settings.path)

    @patch("backend.detector.Path.exists")
    @patch("ultralytics.YOLO")
    def test_yolo_initialization_error(
        self, mock_yolo: Mock, mock_exists: Mock, sample_model_settings: ModelSettings
    ) -> None:
        """Test ModelLoadError when YOLO model initialization fails."""
        mock_exists.return_value = True
        mock_yolo.side_effect = RuntimeError("Invalid model format")

        detector = VehicleDetector(sample_model_settings)

        with pytest.raises(ModelLoadError) as exc_info:
            detector._load_model()

        assert "Failed to load YOLO model" in str(exc_info.value)
        assert exc_info.value.model_path == str(sample_model_settings.path)

    @patch("backend.detector.Path.exists")
    @patch("ultralytics.YOLO")
    def test_model_already_loaded_skip(
        self, mock_yolo: Mock, mock_exists: Mock, sample_model_settings: ModelSettings
    ) -> None:
        """Test that model loading is skipped if already loaded."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        detector = VehicleDetector(sample_model_settings)
        detector._model_loaded = True  # Mark as already loaded

        detector._load_model()

        # Should not call YOLO constructor
        mock_yolo.assert_not_called()

    @patch("backend.detector.Path.exists")
    @patch("ultralytics.YOLO")
    def test_model_warmup_success(
        self, mock_yolo: Mock, mock_exists: Mock, sample_model_settings: ModelSettings
    ) -> None:
        """Test successful model warmup after loading."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.to = Mock()
        # Make the mock callable and return a mock result
        mock_model.return_value = Mock()
        mock_yolo.return_value = mock_model

        detector = VehicleDetector(sample_model_settings)
        detector._load_model()

        # Warmup should have been called during loading
        mock_model.assert_called_once()
        args = mock_model.call_args[0]
        assert isinstance(args[0], np.ndarray)
        assert args[0].shape == (640, 640, 3)

    @patch("backend.detector.Path.exists")
    @patch("ultralytics.YOLO")
    def test_model_warmup_failure_non_critical(
        self, mock_yolo: Mock, mock_exists: Mock, sample_model_settings: ModelSettings
    ) -> None:
        """Test that warmup failure doesn't prevent model loading."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.to = Mock()
        # Make the mock callable but raise error
        mock_model.side_effect = RuntimeError("Warmup failed")
        mock_yolo.return_value = mock_model

        detector = VehicleDetector(sample_model_settings)

        # Should not raise exception despite warmup failure
        detector._load_model()
        assert detector._model_loaded is True

    def test_model_without_to_method(self, sample_model_settings: ModelSettings) -> None:
        """Test model loading when model doesn't have 'to' method."""
        with (
            patch("backend.detector.Path.exists", return_value=True),
            patch("ultralytics.YOLO") as mock_yolo,
        ):
            mock_model = Mock(spec=[])  # Model without 'to' method
            mock_yolo.return_value = mock_model

            detector = VehicleDetector(sample_model_settings)
            detector._load_model()

            assert detector._model_loaded is True
            assert detector._model == mock_model


class TestVehicleDetectorFrameDetection:
    """Test vehicle detection from frames."""

    def test_detect_vehicles_success(
        self,
        sample_model_settings: ModelSettings,
        sample_frame: np.ndarray,
        mock_yolo_results: list[Mock],
    ) -> None:
        """Test successful vehicle detection."""
        detector = VehicleDetector(sample_model_settings)

        with (
            patch.object(detector, "_load_model"),
            patch.object(detector, "_extract_vehicle_detections") as mock_extract,
        ):
            detector._model_loaded = True
            mock_model = Mock(return_value=mock_yolo_results)
            detector._model = mock_model

            mock_detections = [Mock(spec=DetectionResult)]
            mock_extract.return_value = mock_detections

            result = detector.detect_vehicles(sample_frame)

            assert result == mock_detections
            assert detector._frame_counter == 1
            mock_model.assert_called_once_with(
                sample_frame,
                conf=sample_model_settings.confidence_threshold,
                verbose=False,
                device=sample_model_settings.device,
            )

    def test_detect_vehicles_with_timestamp(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test vehicle detection with custom timestamp."""
        detector = VehicleDetector(sample_model_settings)
        custom_timestamp = 1234567890.0

        with (
            patch.object(detector, "_load_model"),
            patch.object(detector, "_extract_vehicle_detections") as mock_extract,
        ):
            detector._model_loaded = True
            detector._model = Mock(return_value=[Mock()])

            detector.detect_vehicles(sample_frame, frame_timestamp=custom_timestamp)

            # Check that custom timestamp was passed to extraction
            mock_extract.assert_called_once()
            args = mock_extract.call_args[0]
            assert args[2] == custom_timestamp  # frame_timestamp argument

    def test_detect_vehicles_lazy_loading(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test that model is loaded lazily on first detection."""
        detector = VehicleDetector(sample_model_settings)

        with (
            patch.object(detector, "_load_model") as mock_load,
            patch.object(detector, "_extract_vehicle_detections"),
        ):
            mock_load.side_effect = lambda: setattr(detector, "_model_loaded", True)
            detector._model = Mock(return_value=[Mock()])

            detector.detect_vehicles(sample_frame)

            mock_load.assert_called_once()

    def test_detect_vehicles_model_load_failure(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test detection failure when model loading fails."""
        detector = VehicleDetector(sample_model_settings)

        with (
            patch.object(detector, "_load_model", side_effect=ModelLoadError("Load failed")),
            pytest.raises(ModelLoadError),
        ):
            detector.detect_vehicles(sample_frame)

    def test_detect_vehicles_model_none_after_loading(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test detection error when model is None after loading."""
        detector = VehicleDetector(sample_model_settings)

        with patch.object(detector, "_load_model"):
            detector._model_loaded = True
            detector._model = None  # Model failed to load properly

            with pytest.raises(DetectionError, match="Model failed to load"):
                detector.detect_vehicles(sample_frame)

    def test_detect_vehicles_model_not_callable(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test detection error when model is not callable."""
        detector = VehicleDetector(sample_model_settings)

        # Create an object that doesn't have __call__
        class NonCallableModel:
            pass

        with patch.object(detector, "_load_model"):
            detector._model_loaded = True
            detector._model = NonCallableModel()  # Object without __call__

            with pytest.raises(DetectionError, match="Model is not callable"):
                detector.detect_vehicles(sample_frame)

    def test_detect_vehicles_inference_exception(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test detection error when inference raises exception."""
        detector = VehicleDetector(sample_model_settings)

        with patch.object(detector, "_load_model"):
            detector._model_loaded = True
            mock_model = Mock()
            mock_model.side_effect = RuntimeError("Inference failed")
            detector._model = mock_model

            with pytest.raises(DetectionError, match="Detection failed"):
                detector.detect_vehicles(sample_frame)

    def test_frame_counter_increments(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test that frame counter increments on each detection."""
        detector = VehicleDetector(sample_model_settings)

        with (
            patch.object(detector, "_load_model"),
            patch.object(detector, "_extract_vehicle_detections", return_value=[]),
        ):
            detector._model_loaded = True
            detector._model = Mock(return_value=[Mock()])

            assert detector._frame_counter == 0
            detector.detect_vehicles(sample_frame)
            assert detector._frame_counter == 1
            detector.detect_vehicles(sample_frame)
            assert detector._frame_counter == 2


class TestVehicleDetectorErrorHandling:
    """Test error handling and validation."""

    def test_validate_frame_success(
        self, sample_model_settings: ModelSettings, sample_frame: np.ndarray
    ) -> None:
        """Test successful frame validation."""
        detector = VehicleDetector(sample_model_settings)

        # Should not raise exception
        detector._validate_frame(sample_frame)

    def test_validate_frame_not_numpy_array(self, sample_model_settings: ModelSettings) -> None:
        """Test frame validation fails for non-numpy array."""
        detector = VehicleDetector(sample_model_settings)

        with pytest.raises(ValueError, match="Frame must be a numpy array"):
            detector._validate_frame([[1, 2, 3]])  # type: ignore

    def test_validate_frame_wrong_dimensions(self, sample_model_settings: ModelSettings) -> None:
        """Test frame validation fails for wrong dimensions."""
        detector = VehicleDetector(sample_model_settings)

        # 2D array instead of 3D
        frame_2d = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame must be 3-dimensional"):
            detector._validate_frame(frame_2d)

        # 4D array
        frame_4d = np.zeros((100, 100, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame must be 3-dimensional"):
            detector._validate_frame(frame_4d)

    def test_validate_frame_wrong_channels(self, sample_model_settings: ModelSettings) -> None:
        """Test frame validation fails for wrong number of channels."""
        detector = VehicleDetector(sample_model_settings)

        # 1 channel (grayscale)
        frame_1ch = np.zeros((100, 100, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame must have 3 channels"):
            detector._validate_frame(frame_1ch)

        # 4 channels (RGBA)
        frame_4ch = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame must have 3 channels"):
            detector._validate_frame(frame_4ch)

    def test_validate_frame_empty(self, sample_model_settings: ModelSettings) -> None:
        """Test frame validation fails for empty frame."""
        detector = VehicleDetector(sample_model_settings)

        empty_frame = np.array([]).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="Frame cannot be empty"):
            detector._validate_frame(empty_frame)

    def test_validate_frame_too_small(self, sample_model_settings: ModelSettings) -> None:
        """Test frame validation fails for too small frame."""
        detector = VehicleDetector(sample_model_settings)

        small_frame = np.zeros((5, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame too small"):
            detector._validate_frame(small_frame)

    def test_validate_frame_too_large(self, sample_model_settings: ModelSettings) -> None:
        """Test frame validation fails for too large frame."""
        detector = VehicleDetector(sample_model_settings)

        large_frame = np.zeros((9000, 9000, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame too large"):
            detector._validate_frame(large_frame)

    def test_detection_error_with_frame_shape(self) -> None:
        """Test DetectionError includes frame shape information."""
        frame_shape = (480, 640, 3)
        error = DetectionError("Test error", frame_shape=frame_shape)

        assert str(error) == "Test error"
        assert error.frame_shape == frame_shape

    def test_model_load_error_with_path(self) -> None:
        """Test ModelLoadError includes model path information."""
        model_path = "/path/to/model.pt"
        error = ModelLoadError("Load failed", model_path=model_path)

        assert str(error) == "Load failed"
        assert error.model_path == model_path


class TestVehicleDetectorUtilities:
    """Test utility functions and information methods."""

    def test_get_model_info_before_loading(self, sample_model_settings: ModelSettings) -> None:
        """Test get_model_info before model is loaded."""
        detector = VehicleDetector(sample_model_settings)

        info = detector.get_model_info()

        expected_info = {
            "model_path": str(sample_model_settings.path),
            "device": sample_model_settings.device,
            "confidence_threshold": sample_model_settings.confidence_threshold,
            "model_loaded": False,
            "frames_processed": 0,
            "supported_vehicle_classes": [1, 2, 3, 5, 7],
        }
        assert info == expected_info

    def test_get_model_info_after_loading(self, sample_model_settings: ModelSettings) -> None:
        """Test get_model_info after model is loaded."""
        detector = VehicleDetector(sample_model_settings)

        # Mock loaded model
        detector._model_loaded = True
        detector._frame_counter = 5
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.names = {0: "person", 1: "bicycle", 2: "car"}
        detector._model = mock_model

        info = detector.get_model_info()

        assert info["model_loaded"] is True
        assert info["frames_processed"] == 5
        assert info["model_type"] == "Mock"
        assert info["class_names"] == {0: "person", 1: "bicycle", 2: "car"}

    def test_get_model_info_model_attributes_error(
        self, sample_model_settings: ModelSettings
    ) -> None:
        """Test get_model_info handles errors getting model attributes gracefully."""
        detector = VehicleDetector(sample_model_settings)

        # Mock loaded model that doesn't have model or names attributes
        detector._model_loaded = True

        # Create a mock object without model or names attributes
        class MockModelWithoutAttrs:
            pass

        detector._model = MockModelWithoutAttrs()

        info = detector.get_model_info()

        # Should not include model_type or class_names due to missing attributes
        assert "model_type" not in info
        assert "class_names" not in info
        assert info["model_loaded"] is True

    def test_reset_frame_counter(self, sample_model_settings: ModelSettings) -> None:
        """Test frame counter reset functionality."""
        detector = VehicleDetector(sample_model_settings)

        detector._frame_counter = 42
        detector.reset_frame_counter()

        assert detector._frame_counter == 0

    def test_extract_vehicle_detections_success(self, sample_model_settings: ModelSettings) -> None:
        """Test successful extraction of vehicle detections."""
        detector = VehicleDetector(sample_model_settings)

        # Mock YOLO results
        mock_boxes = Mock()
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu = Mock()
        mock_boxes.xyxy.cpu().numpy = Mock(return_value=np.array([[10, 20, 50, 80]]))
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu = Mock()
        mock_boxes.conf.cpu().numpy = Mock(return_value=np.array([0.9]))
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu = Mock()
        mock_boxes.cls.cpu().numpy = Mock(return_value=np.array([2]))  # car class

        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_yolo_results = [mock_result]

        frame_shape = (480, 640, 3)
        frame_timestamp = 1234567890.0
        frame_id = 1

        with (
            patch("backend.detector.map_coco_class_to_vehicle_type", return_value=VehicleType.CAR),
            patch("backend.detector.DetectionResult.from_yolo_detection") as mock_from_yolo,
        ):
            mock_detection = Mock(spec=DetectionResult)
            mock_from_yolo.return_value = mock_detection

            detections = detector._extract_vehicle_detections(
                mock_yolo_results, frame_shape, frame_timestamp, frame_id
            )

            assert len(detections) == 1
            assert detections[0] == mock_detection

            mock_from_yolo.assert_called_once_with(
                x1=10,
                y1=20,
                x2=50,
                y2=80,
                confidence=0.9,
                class_id=2,
                vehicle_type=VehicleType.CAR,
                frame_timestamp=frame_timestamp,
                frame_id=frame_id,
                frame_shape=frame_shape,
            )

    def test_extract_vehicle_detections_filters_non_vehicles(
        self, sample_model_settings: ModelSettings
    ) -> None:
        """Test that non-vehicle detections are filtered out."""
        detector = VehicleDetector(sample_model_settings)

        # Mock YOLO results with person class (not a vehicle)
        mock_boxes = Mock()
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu = Mock()
        mock_boxes.xyxy.cpu().numpy = Mock(return_value=np.array([[10, 20, 50, 80]]))
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu = Mock()
        mock_boxes.conf.cpu().numpy = Mock(return_value=np.array([0.9]))
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu = Mock()
        mock_boxes.cls.cpu().numpy = Mock(return_value=np.array([0]))  # person class

        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_yolo_results = [mock_result]

        detections = detector._extract_vehicle_detections(
            mock_yolo_results, (480, 640, 3), 1234567890.0, 1
        )

        assert len(detections) == 0

    def test_extract_vehicle_detections_no_boxes(
        self, sample_model_settings: ModelSettings
    ) -> None:
        """Test extraction when YOLO result has no boxes."""
        detector = VehicleDetector(sample_model_settings)

        mock_result = Mock()
        mock_result.boxes = None
        mock_yolo_results = [mock_result]

        detections = detector._extract_vehicle_detections(
            mock_yolo_results, (480, 640, 3), 1234567890.0, 1
        )

        assert len(detections) == 0

    def test_extract_vehicle_detections_exception_handling(
        self, sample_model_settings: ModelSettings
    ) -> None:
        """Test that extraction exceptions are handled gracefully."""
        detector = VehicleDetector(sample_model_settings)

        # Mock YOLO results that raise exception
        mock_yolo_results = Mock(side_effect=AttributeError("No boxes attribute"))

        detections = detector._extract_vehicle_detections(
            mock_yolo_results, (480, 640, 3), 1234567890.0, 1
        )

        # Should return empty list instead of crashing
        assert len(detections) == 0


# Fixtures
@pytest.fixture
def sample_model_settings() -> ModelSettings:
    """Create sample ModelSettings for testing."""
    return ModelSettings(path="models/yolo11n.pt", device="cpu", confidence_threshold=0.5)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create sample frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_yolo_results() -> list[Mock]:
    """Create mock YOLO results for testing."""
    mock_boxes = Mock()
    mock_boxes.xyxy = Mock()
    mock_boxes.xyxy.cpu = Mock()
    mock_boxes.xyxy.cpu().numpy = Mock(return_value=np.array([[10, 20, 50, 80]]))
    mock_boxes.conf = Mock()
    mock_boxes.conf.cpu = Mock()
    mock_boxes.conf.cpu().numpy = Mock(return_value=np.array([0.9]))
    mock_boxes.cls = Mock()
    mock_boxes.cls.cpu = Mock()
    mock_boxes.cls.cpu().numpy = Mock(return_value=np.array([2]))

    mock_result = Mock()
    mock_result.boxes = mock_boxes

    return [mock_result]
