"""Vehicle detection module using YOLO models.

This module provides the VehicleDetector class for real-time vehicle detection
from camera frames using ultralytics YOLO models. It integrates with the
TrafficMetry configuration system and provides filtered detections based on
COCO vehicle classes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from backend.config import ModelSettings
from backend.detection_models import (
    DetectionResult,
    map_coco_class_to_vehicle_type,
)

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Raised when vehicle detection fails."""

    def __init__(self, message: str, frame_shape: tuple | None = None) -> None:
        """Initialize detection error.

        Args:
            message: Error description
            frame_shape: Shape of frame that caused the error, if available
        """
        super().__init__(message)
        self.frame_shape = frame_shape


class ModelLoadError(DetectionError):
    """Raised when YOLO model fails to load."""

    def __init__(self, message: str, model_path: str | None = None) -> None:
        """Initialize model load error.

        Args:
            message: Error description
            model_path: Path to model that failed to load, if available
        """
        super().__init__(message)
        self.model_path = model_path


class VehicleDetector:
    """AI-powered vehicle detection using YOLO models.

    This class provides vehicle detection capabilities using ultralytics YOLO
    models with lazy loading, filtering by vehicle types, and integration with
    the TrafficMetry configuration system.

    The detector filters YOLO detections to only include vehicle classes:
    - car (COCO class 2)
    - motorcycle (COCO class 3)
    - bus (COCO class 5)
    - truck (COCO class 7)
    - bicycle (COCO class 1)

    Example:
        ```python
        from backend.config import get_config

        config = get_config()
        detector = VehicleDetector(config.model)

        detections = detector.detect_vehicles(frame)
        for detection in detections:
            print(f"Found {detection.vehicle_type} at {detection.centroid}")
        ```
    """

    # COCO class IDs for vehicle types we want to detect
    VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

    def __init__(self, model_settings: ModelSettings) -> None:
        """Initialize VehicleDetector.

        Args:
            model_settings: Configuration for YOLO model parameters

        Raises:
            ValueError: If model_settings is invalid
        """
        self.model_settings = model_settings
        self._model: object | None = None  # YOLO model (lazy loaded)
        self._model_loaded = False
        self._frame_counter = 0

        # Validate model settings
        if not isinstance(model_settings.path, str | Path):
            raise ValueError("Model path must be a string or Path object")

        if not 0.0 <= model_settings.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        logger.info(f"VehicleDetector initialized with model: {model_settings.path}")

    def _load_model(self) -> None:
        """Load YOLO model with error handling.

        Raises:
            ModelLoadError: If model fails to load
        """
        if self._model_loaded:
            return

        try:
            # Import here to avoid dependency issues during testing
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model from: {self.model_settings.path}")
            start_time = time.time()

            # Validate model path exists
            model_path = Path(self.model_settings.path)
            if not model_path.exists():
                raise ModelLoadError(
                    f"Model file not found: {model_path}", model_path=str(model_path)
                )

            # Load model
            self._model = YOLO(str(model_path))

            # Configure model device (only for PyTorch models, not for exported formats)
            # OpenVINO and other exported formats don't support .to() method
            # Device should be specified in predict() calls instead
            if hasattr(self._model, "to") and callable(self._model):
                self._model.to(self.model_settings.device)

            load_time = time.time() - start_time
            logger.info(
                f"YOLO model loaded successfully in {load_time:.2f}s on device: {self.model_settings.device}"
            )

            # Perform warmup inference if possible
            self._warmup_model()

            self._model_loaded = True

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import ultralytics YOLO: {e}. Install with: pip install ultralytics",
                model_path=str(self.model_settings.path),
            ) from e

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YOLO model: {e}", model_path=str(self.model_settings.path)
            ) from e

    def _warmup_model(self) -> None:
        """Perform model warmup with dummy inference.

        This helps optimize inference speed for subsequent real detections.
        """
        try:
            if self._model is None:
                return

            logger.debug("Performing model warmup inference...")

            # Create dummy frame for warmup
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

            # Perform warmup inference with format detection
            if hasattr(self._model, "predict"):
                # OpenVINO format - use predict() method
                _ = self._model.predict(dummy_frame, verbose=False)
            elif callable(self._model):
                # PyTorch format - use callable
                _ = self._model(dummy_frame, verbose=False)

            logger.debug("Model warmup completed successfully")

        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

    def detect_vehicles(
        self,
        frame: NDArray,
        frame_timestamp: float | None = None,
    ) -> list[DetectionResult]:
        """Detect vehicles in the provided frame.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR format
            frame_timestamp: Unix timestamp for the frame, uses current time if None

        Returns:
            List of DetectionResult objects for detected vehicles

        Raises:
            DetectionError: If detection fails
            ValueError: If frame is invalid
        """
        # Validate input frame
        self._validate_frame(frame)

        # Use current time if timestamp not provided
        if frame_timestamp is None:
            frame_timestamp = time.time()

        # Increment frame counter
        self._frame_counter += 1

        try:
            # Load model if not already loaded (lazy loading)
            if not self._model_loaded:
                self._load_model()

            if self._model is None:
                raise DetectionError("Model failed to load", frame_shape=frame.shape)

            # Perform inference
            logger.debug(
                f"Running inference on frame {self._frame_counter} with shape {frame.shape}"
            )

            # OpenVINO models use .predict() method, PyTorch models are callable
            if hasattr(self._model, "predict"):
                # OpenVINO format - use predict() method
                results = self._model.predict(
                    frame,
                    conf=self.model_settings.confidence_threshold,
                    verbose=False,
                    device=self.model_settings.device,
                )
            elif callable(self._model):
                # PyTorch format - use callable
                results = self._model(
                    frame,
                    conf=self.model_settings.confidence_threshold,
                    verbose=False,
                    device=self.model_settings.device,
                )
            else:
                raise DetectionError(
                    "Model is not callable and has no predict method", frame_shape=frame.shape
                )

            # Extract and filter detections
            detections = self._extract_vehicle_detections(
                results, frame.shape, frame_timestamp, self._frame_counter
            )

            logger.debug(
                f"Found {len(detections)} vehicle detections in frame {self._frame_counter}"
            )

            return detections

        except ModelLoadError:
            # Re-raise model loading errors
            raise

        except Exception as e:
            raise DetectionError(
                f"Detection failed for frame {self._frame_counter}: {e}", frame_shape=frame.shape
            ) from e

    def _validate_frame(self, frame: NDArray) -> None:
        """Validate input frame format and dimensions.

        Args:
            frame: Input frame to validate

        Raises:
            ValueError: If frame format is invalid
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")

        if frame.ndim != 3:
            raise ValueError(f"Frame must be 3-dimensional (H, W, C), got {frame.ndim}D")

        if frame.shape[2] != 3:
            raise ValueError(f"Frame must have 3 channels (BGR), got {frame.shape[2]}")

        if frame.size == 0:
            raise ValueError("Frame cannot be empty")

        # Check reasonable size limits
        height, width, _ = frame.shape
        if height < 10 or width < 10:
            raise ValueError(f"Frame too small: {width}x{height}, minimum 10x10")

        if height > 8192 or width > 8192:
            raise ValueError(f"Frame too large: {width}x{height}, maximum 8192x8192")

    def _extract_vehicle_detections(
        self,
        yolo_results: object,
        frame_shape: tuple,
        frame_timestamp: float,
        frame_id: int,
    ) -> list[DetectionResult]:
        """Extract vehicle detections from YOLO results.

        Args:
            yolo_results: YOLO model inference results
            frame_shape: Shape of the input frame (H, W, C)
            frame_timestamp: Unix timestamp for the frame
            frame_id: Sequential frame identifier

        Returns:
            List of DetectionResult objects for vehicles only
        """
        detections: list[DetectionResult] = []

        try:
            # Handle YOLO results format (ultralytics)
            if hasattr(yolo_results, "__iter__"):
                for result in yolo_results:
                    if hasattr(result, "boxes") and result.boxes is not None:
                        boxes = result.boxes

                        # Extract box data
                        if (
                            hasattr(boxes, "xyxy")
                            and hasattr(boxes, "conf")
                            and hasattr(boxes, "cls")
                        ):
                            coords = boxes.xyxy.cpu().numpy()  # (N, 4) - x1, y1, x2, y2
                            confidences = boxes.conf.cpu().numpy()  # (N,)
                            class_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

                            # Process each detection
                            for coord, confidence, class_id in zip(
                                coords, confidences, class_ids, strict=False
                            ):
                                # Filter for vehicle classes only
                                if class_id not in self.VEHICLE_CLASS_IDS:
                                    continue

                                # Convert coordinates to integers
                                x1, y1, x2, y2 = map(int, coord)

                                # Map COCO class to vehicle type
                                vehicle_type = map_coco_class_to_vehicle_type(class_id)

                                # Create detection result
                                detection = DetectionResult.from_yolo_detection(
                                    x1=x1,
                                    y1=y1,
                                    x2=x2,
                                    y2=y2,
                                    confidence=float(confidence),
                                    class_id=class_id,
                                    vehicle_type=vehicle_type,
                                    frame_timestamp=frame_timestamp,
                                    frame_id=frame_id,
                                    frame_shape=frame_shape,
                                )

                                detections.append(detection)

        except Exception as e:
            logger.error(f"Failed to extract detections from YOLO results: {e}")
            # Return empty list rather than crashing
            return []

        return detections

    def get_model_info(self) -> dict[str, object]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": str(self.model_settings.path),
            "device": self.model_settings.device,
            "confidence_threshold": self.model_settings.confidence_threshold,
            "model_loaded": self._model_loaded,
            "frames_processed": self._frame_counter,
            "supported_vehicle_classes": list(self.VEHICLE_CLASS_IDS),
        }

        if self._model_loaded and self._model is not None:
            try:
                # Try to get additional model info if available
                if hasattr(self._model, "model"):
                    info["model_type"] = str(type(self._model.model).__name__)
                if hasattr(self._model, "names"):
                    info["class_names"] = self._model.names
            except Exception:
                # Ignore errors getting additional info
                pass

        return info

    def reset_frame_counter(self) -> None:
        """Reset the internal frame counter.

        Useful for testing or when starting a new detection session.
        """
        self._frame_counter = 0
        logger.debug("Frame counter reset to 0")

    def __repr__(self) -> str:
        """String representation of VehicleDetector."""
        return (
            f"VehicleDetector("
            f"model_path='{self.model_settings.path}', "
            f"device='{self.model_settings.device}', "
            f"confidence={self.model_settings.confidence_threshold}, "
            f"loaded={self._model_loaded})"
        )
