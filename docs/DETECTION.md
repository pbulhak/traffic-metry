# AI Detection Module for TrafficMetry

This guide explains how to use the AI detection module to detect vehicles in video streams using YOLO models. The module provides real-time vehicle detection with support for filtering by vehicle types and integrates seamlessly with the TrafficMetry monitoring system.

## Overview

The TrafficMetry detection module provides:
- **Real-time vehicle detection** using ultralytics YOLO models
- **Vehicle type classification** for cars, trucks, buses, motorcycles, and bicycles
- **Lazy loading** of AI models for efficient resource usage
- **Comprehensive error handling** with detailed diagnostics
- **Performance optimization** with model warmup and frame validation
- **Integration** with camera streams and configuration management

## Architecture

### Core Components

**VehicleDetector Class** (`backend/detector.py`)
- Main detection engine with YOLO model integration
- Lazy loading pattern for efficient memory usage
- Comprehensive frame validation and error handling
- Performance monitoring with frame counters and diagnostics

**Detection Models** (`backend/detection_models.py`)
- `DetectionResult` - represents a single vehicle detection
- `VehicleType` - enumeration of supported vehicle types
- COCO class mapping utilities for YOLO integration

**Custom Exceptions**
- `DetectionError` - general detection failures with frame context
- `ModelLoadError` - YOLO model loading failures with path information

## Quick Start

### Basic Usage

```python
from backend.config import get_config
from backend.detector import VehicleDetector

# Load configuration
config = get_config()

# Initialize detector with model settings
detector = VehicleDetector(config.model)

# Detect vehicles in a frame
detections = detector.detect_vehicles(frame)

# Process results
for detection in detections:
    print(f"Found {detection.vehicle_type} at {detection.centroid}")
    print(f"Confidence: {detection.confidence:.2f}")
    print(f"Bounding box: ({detection.x1}, {detection.y1}) to ({detection.x2}, {detection.y2})")
```

### Integration with Camera Stream

```python
from backend.camera_stream import CameraStream
from backend.detector import VehicleDetector

# Initialize components
camera = CameraStream(config.camera)
detector = VehicleDetector(config.model)

try:
    with camera:
        while True:
            # Capture frame from camera
            frame = camera.get_frame()
            if frame is None:
                continue
                
            # Detect vehicles
            detections = detector.detect_vehicles(frame)
            
            # Process detections
            for detection in detections:
                # Your processing logic here
                pass
                
except KeyboardInterrupt:
    print("Detection stopped")
```

## Configuration

### Model Settings

Configure the YOLO model in your `.env` file:

```bash
# Model Configuration
MODEL__PATH=models/yolov8n.pt
MODEL__CONFIDENCE_THRESHOLD=0.6
MODEL__DEVICE=cpu
MODEL__MAX_DETECTIONS=50
```

**Configuration Options:**
- `MODEL__PATH` - Path to YOLO model file (.pt format)
- `MODEL__CONFIDENCE_THRESHOLD` - Minimum confidence score (0.0-1.0)
- `MODEL__DEVICE` - Processing device ("cpu", "cuda", "mps")
- `MODEL__MAX_DETECTIONS` - Maximum detections per frame

### Supported Vehicle Types

The detection module filters YOLO detections to include only vehicle classes:

| Vehicle Type | COCO Class ID | Description |
|-------------|---------------|-------------|
| `car` | 2 | Passenger cars, sedans, hatchbacks |
| `motorcycle` | 3 | Motorcycles, scooters, bikes with engines |
| `bus` | 5 | Public buses, coaches, large passenger vehicles |
| `truck` | 7 | Cargo trucks, delivery vehicles, semi-trailers |
| `bicycle` | 1 | Pedal bicycles, e-bikes |

## API Reference

### VehicleDetector Class

#### Constructor

```python
def __init__(self, model_settings: ModelSettings) -> None
```

Initialize the vehicle detector with model configuration.

**Parameters:**
- `model_settings` - ModelSettings object with YOLO configuration

**Raises:**
- `ValueError` - If model settings are invalid

#### detect_vehicles()

```python
def detect_vehicles(
    self, 
    frame: NDArray, 
    frame_timestamp: float | None = None
) -> list[DetectionResult]
```

Detect vehicles in the provided frame.

**Parameters:**
- `frame` - Input image as numpy array (H, W, C) in BGR format
- `frame_timestamp` - Unix timestamp for the frame (optional)

**Returns:**
- List of `DetectionResult` objects for detected vehicles

**Raises:**
- `DetectionError` - If detection fails
- `ValueError` - If frame format is invalid

#### get_model_info()

```python
def get_model_info(self) -> dict[str, object]
```

Get information about the loaded model and detector state.

**Returns:**
- Dictionary with model information including:
  - `model_path` - Path to loaded model
  - `device` - Processing device
  - `confidence_threshold` - Detection threshold
  - `model_loaded` - Loading status
  - `frames_processed` - Number of processed frames

#### reset_frame_counter()

```python
def reset_frame_counter(self) -> None
```

Reset the internal frame counter for new detection sessions.

### DetectionResult Class

#### Properties

```python
# Bounding box coordinates (pixels)
x1: int          # Left edge
y1: int          # Top edge
x2: int          # Right edge
y2: int          # Bottom edge

# Detection metadata
confidence: float              # Detection confidence (0.0-1.0)
class_id: int                 # COCO class identifier
vehicle_type: VehicleType     # Classified vehicle type
detection_id: str             # Unique detection identifier
frame_timestamp: float        # Frame capture timestamp
frame_id: int                # Sequential frame number

# Computed properties
@property
def centroid(self) -> tuple[int, int]     # Center point
@property
def bbox_area_pixels(self) -> int         # Area in pixels
@property
def bbox_width(self) -> int               # Width in pixels
@property
def bbox_height(self) -> int              # Height in pixels
@property
def aspect_ratio(self) -> float           # Width/height ratio
```

#### Factory Method

```python
@classmethod
def from_yolo_detection(
    cls,
    x1: int, y1: int, x2: int, y2: int,
    confidence: float,
    class_id: int,
    vehicle_type: VehicleType,
    frame_timestamp: float,
    frame_id: int,
    frame_shape: tuple,
    detection_id: str | None = None,
) -> DetectionResult
```

Create DetectionResult from YOLO model output with automatic ID generation.

## Advanced Usage

### Custom Model Loading

```python
from pathlib import Path

# Custom model configuration
custom_settings = ModelSettings(
    path=Path("models/custom_yolov8s.pt"),
    confidence_threshold=0.7,
    device="cuda",
    max_detections=100
)

detector = VehicleDetector(custom_settings)
```

### Error Handling

```python
from backend.detector import DetectionError, ModelLoadError

try:
    detector = VehicleDetector(model_settings)
    detections = detector.detect_vehicles(frame)
    
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
    print(f"Model path: {e.model_path}")
    
except DetectionError as e:
    print(f"Detection failed: {e}")
    print(f"Frame shape: {e.frame_shape}")
    
except ValueError as e:
    print(f"Invalid frame: {e}")
```

### Performance Monitoring

```python
# Get detector diagnostics
info = detector.get_model_info()
print(f"Processed {info['frames_processed']} frames")
print(f"Model loaded: {info['model_loaded']}")
print(f"Device: {info['device']}")

# Monitor detection performance
import time

start_time = time.time()
detections = detector.detect_vehicles(frame)
detection_time = time.time() - start_time

print(f"Detection took {detection_time:.3f}s for {len(detections)} vehicles")
```

### Batch Processing

```python
import numpy as np
from pathlib import Path

def process_video_file(video_path: Path, output_path: Path):
    """Process video file and save detection results."""
    
    detector = VehicleDetector(config.model)
    results = []
    
    # Process video frames (pseudo-code)
    for frame_id, frame in enumerate(video_frames(video_path)):
        detections = detector.detect_vehicles(frame, frame_timestamp=time.time())
        
        # Save results
        for detection in detections:
            results.append({
                'frame_id': frame_id,
                'vehicle_type': detection.vehicle_type,
                'confidence': detection.confidence,
                'bbox': [detection.x1, detection.y1, detection.x2, detection.y2],
                'centroid': detection.centroid
            })
    
    # Export results
    save_detection_results(results, output_path)
    
    print(f"Processed {detector.get_model_info()['frames_processed']} frames")
```

## Performance Optimization

### Model Selection

**Recommended models by use case:**

| Use Case | Model | Speed | Accuracy | Memory |
|----------|-------|-------|----------|---------|
| Real-time monitoring | YOLOv8n | Fast | Good | Low |
| Balanced performance | YOLOv8s | Medium | Better | Medium |
| High accuracy analysis | YOLOv8m | Slow | Best | High |

### Hardware Configuration

**CPU Optimization:**
- Use YOLOv8n for real-time performance
- Set `MODEL__DEVICE=cpu`
- Consider frame downscaling for faster processing

**GPU Optimization:**
- Install CUDA-compatible PyTorch
- Set `MODEL__DEVICE=cuda`
- Use larger models (YOLOv8s/m) for better accuracy
- Enable model warmup for consistent performance

### Frame Processing

```python
import cv2

# Optimize frame preprocessing
def preprocess_frame(frame, target_size=(640, 640)):
    """Resize frame for optimal YOLO performance."""
    return cv2.resize(frame, target_size)

# Use optimized detection
frame = camera.get_frame()
if frame is not None:
    # Resize for faster processing
    processed_frame = preprocess_frame(frame)
    detections = detector.detect_vehicles(processed_frame)
```

## Troubleshooting

### Common Issues

| Problem | Symptoms | Solution |
|---------|----------|----------|
| Model not loading | `ModelLoadError` on startup | Check model path exists, install ultralytics |
| Poor detection accuracy | Missing vehicles or false positives | Adjust confidence threshold, try different model |
| Slow performance | High detection latency | Use smaller model, resize frames, check device settings |
| Memory errors | Out of memory crashes | Reduce max detections, use CPU device, smaller frames |

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('backend.detector')

# Detector will now log detailed information
detector = VehicleDetector(model_settings)
```

### Validation

Verify your detection setup:

```python
# Test detector initialization
try:
    detector = VehicleDetector(config.model)
    info = detector.get_model_info()
    print("✅ Detector initialized successfully")
    print(f"Model: {info['model_path']}")
    print(f"Device: {info['device']}")
    
except Exception as e:
    print(f"❌ Detector initialization failed: {e}")

# Test frame detection
test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
try:
    detections = detector.detect_vehicles(test_frame)
    print(f"✅ Detection test passed: {len(detections)} vehicles found")
except Exception as e:
    print(f"❌ Detection test failed: {e}")
```

## Integration Examples

### TrafficMetry Pipeline

```python
from backend.camera_stream import CameraStream
from backend.detector import VehicleDetector
from backend.config import get_config

def main_detection_loop():
    """Main TrafficMetry detection pipeline."""
    
    config = get_config()
    
    # Initialize components
    camera = CameraStream(config.camera)
    detector = VehicleDetector(config.model)
    
    print(f"Starting detection with model: {detector.get_model_info()['model_path']}")
    
    try:
        with camera:
            while True:
                # Capture frame
                frame = camera.get_frame()
                if frame is None:
                    continue
                
                # Detect vehicles
                detections = detector.detect_vehicles(frame)
                
                # Process each detection
                for detection in detections:
                    # Convert to API v2.3 format
                    vehicle_event = {
                        "eventId": detection.detection_id,
                        "timestamp": detection.frame_timestamp,
                        "vehicleType": detection.vehicle_type,
                        "position": {
                            "boundingBox": {
                                "x1": detection.x1,
                                "y1": detection.y1,
                                "x2": detection.x2,
                                "y2": detection.y2
                            }
                        },
                        "analytics": {
                            "confidence": detection.confidence
                        }
                    }
                    
                    # Send to processing pipeline
                    process_vehicle_event(vehicle_event)
                
    except KeyboardInterrupt:
        print("Detection stopped by user")
    
    except Exception as e:
        print(f"Detection error: {e}")

if __name__ == "__main__":
    main_detection_loop()
```

### Custom Detection Pipeline

```python
class CustomDetectionPipeline:
    """Custom detection pipeline with filtering and analytics."""
    
    def __init__(self, model_settings):
        self.detector = VehicleDetector(model_settings)
        self.detection_history = []
        
    def process_frame(self, frame, min_confidence=0.7):
        """Process frame with custom filtering."""
        
        # Get raw detections
        detections = self.detector.detect_vehicles(frame)
        
        # Apply custom filters
        filtered_detections = [
            d for d in detections 
            if d.confidence >= min_confidence
            and d.bbox_area_pixels > 1000  # Minimum size filter
        ]
        
        # Store in history
        self.detection_history.extend(filtered_detections)
        
        # Analytics
        vehicle_counts = {}
        for detection in filtered_detections:
            vehicle_type = detection.vehicle_type
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
            
        return {
            'detections': filtered_detections,
            'counts': vehicle_counts,
            'total': len(filtered_detections)
        }
    
    def get_statistics(self):
        """Get detection statistics."""
        if not self.detection_history:
            return {}
            
        total_detections = len(self.detection_history)
        avg_confidence = sum(d.confidence for d in self.detection_history) / total_detections
        
        type_distribution = {}
        for detection in self.detection_history:
            vehicle_type = detection.vehicle_type
            type_distribution[vehicle_type] = type_distribution.get(vehicle_type, 0) + 1
            
        return {
            'total_detections': total_detections,
            'average_confidence': avg_confidence,
            'vehicle_distribution': type_distribution,
            'frames_processed': self.detector.get_model_info()['frames_processed']
        }
```

## Support

For additional help with the AI detection module:

- Check the main [README.md](../README.md) for system requirements
- Review [MCP.md](../MCP.md) for technical specifications
- See [CALIBRATION.md](CALIBRATION.md) for camera calibration
- Submit issues to the project repository
- Consult system logs for detailed error messages

---

*This documentation is part of the TrafficMetry traffic monitoring system. For complete system documentation, see the project documentation in the `docs/` directory.*