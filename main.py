"""TrafficMetry Main Application - Vehicle Detection and Analysis System.

This is the main entry point for the TrafficMetry system that orchestrates
all components to provide real-time vehicle detection and analysis.

The application integrates:
- Camera stream management for live video capture
- AI-powered vehicle detection using YOLO models
- Lane assignment based on camera calibration data
- Event generation compatible with API v2.3 format
- Candidate image saving for future model training
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.camera_stream import CameraConnectionError, CameraStream
from backend.config import Config, get_config
from backend.detector import DetectionError, ModelLoadError, VehicleDetector
from backend.detection_models import DetectionResult, VehicleType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trafficmetry.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class LaneAnalyzer:
    """Analyzes vehicle positions and assigns lanes based on calibration data."""
    
    def __init__(self, lanes_config: Optional[object]) -> None:
        """Initialize lane analyzer with calibration configuration.
        
        Args:
            lanes_config: Lane configuration from calibration, or None if not calibrated
        """
        self.lanes_config = lanes_config
        self.lane_boundaries: List[int] = []
        self.lane_directions: Dict[int, str] = {}
        
        if lanes_config and hasattr(lanes_config, 'lines') and hasattr(lanes_config, 'directions'):
            self._calculate_lane_boundaries()
            self.lane_directions = dict(lanes_config.directions)
            logger.info(f"Lane analyzer initialized with {len(self.lane_boundaries)-1} lanes")
        else:
            logger.warning("No valid calibration data - lane assignment will be disabled")
    
    def _calculate_lane_boundaries(self) -> None:
        """Calculate lane boundaries from calibration lines.
        
        For horizontal lanes, we use Y-coordinates of the lines to define boundaries.
        Lines are sorted by Y-coordinate to create lane regions.
        """
        if not self.lanes_config or not hasattr(self.lanes_config, 'lines'):
            return
            
        # Extract Y-coordinates from calibration lines (horizontal lanes)
        y_coordinates = [
            (y1 + y2) // 2 for _, y1, _, y2 in self.lanes_config.lines
        ]
        
        # Sort Y-coordinates to create lane boundaries
        self.lane_boundaries = sorted(y_coordinates)
        logger.debug(f"Lane boundaries calculated: {self.lane_boundaries}")
    
    def assign_lane(self, detection: DetectionResult) -> Tuple[Optional[int], Optional[str]]:
        """Assign lane number and direction to a vehicle detection.
        
        Args:
            detection: Vehicle detection result
            
        Returns:
            Tuple of (lane_number, direction) or (None, None) if assignment fails
        """
        if not self.lane_boundaries:
            logger.debug("No lane boundaries available - returning None")
            return None, None
        
        # Get vehicle centroid Y-coordinate (horizontal lanes)
        vehicle_y = detection.centroid[1]
        
        # Find which lane the vehicle belongs to
        lane_number = self._find_lane_for_y_coordinate(vehicle_y)
        
        if lane_number is not None:
            direction = self.lane_directions.get(lane_number, "stationary")
            logger.debug(f"Vehicle at Y={vehicle_y} assigned to lane {lane_number}, direction {direction}")
            return lane_number, direction
        
        logger.debug(f"Vehicle at Y={vehicle_y} could not be assigned to any lane")
        return None, None
    
    def _find_lane_for_y_coordinate(self, y: int) -> Optional[int]:
        """Find lane number for given Y coordinate.
        
        Args:
            y: Y-coordinate of vehicle centroid
            
        Returns:
            Lane number (0-based) or None if outside all lanes
        """
        if len(self.lane_boundaries) < 2:
            return None

        for i in range(len(self.lane_boundaries) - 1):
            top_y = self.lane_boundaries[i]
            bottom_y = self.lane_boundaries[i + 1]
            if top_y <= y < bottom_y:
                return i  # Zwraca indeks pasa (0, 1, ...)
        
        return None  # Pojazd jest poza wszystkimi zdefiniowanymi pasami


class EventGenerator:
    """Generates API v2.3 compatible events from detection results."""
    
    def __init__(self) -> None:
        """Initialize event generator."""
        self.event_counter = 0
    
    def create_vehicle_event(
        self,
        detection: DetectionResult,
        lane_number: Optional[int],
        direction: Optional[str]
    ) -> Dict:
        """Create API v2.3 compatible vehicle event.
        
        Args:
            detection: Vehicle detection result
            lane_number: Assigned lane number (None if unassigned)
            direction: Movement direction (None if unknown)
            
        Returns:
            Dictionary representing vehicle event in API v2.3 format
        """
        self.event_counter += 1
        
        # Generate unique vehicle ID (combination of detection ID and counter)
        vehicle_id = f"{detection.detection_id}-{self.event_counter}"
        
        event = {
            "eventId": str(uuid.uuid4()),
            "timestamp": detection.frame_timestamp,
            "vehicleId": vehicle_id,
            "vehicleType": detection.vehicle_type.value,
            "movement": {
                "direction": direction or "stationary",
                "lane": lane_number if lane_number is not None else -1
            },
            "vehicleColor": {
                "hex": None,  # Will be implemented in Phase 3
                "name": None
            },
            "position": {
                "boundingBox": {
                    "x1": detection.x1,
                    "y1": detection.y1,
                    "x2": detection.x2,
                    "y2": detection.y2
                }
            },
            "analytics": {
                "confidence": detection.confidence,
                "estimatedSpeedKph": None  # Future implementation
            }
        }
        
        return event


class CandidateSaver:
    """Saves detected vehicle images as training candidates."""
    
    def __init__(self, output_dir: Path = Path("data/unlabeled_images")) -> None:
        """Initialize candidate saver.
        
        Args:
            output_dir: Directory to save candidate images
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0
        
        logger.info(f"Candidate saver initialized - output directory: {self.output_dir}")
    
    def save_candidate(self, frame: NDArray, detection: DetectionResult) -> Optional[Path]:
        """Save vehicle detection as candidate image.
        
        Args:
            frame: Source video frame
            detection: Vehicle detection result
            
        Returns:
            Path to saved image or None if saving failed
        """
        try:
            # Extract vehicle region from frame
            vehicle_crop = frame[detection.y1:detection.y2, detection.x1:detection.x2]
            
            if vehicle_crop.size == 0:
                logger.warning(f"Empty crop for detection {detection.detection_id}")
                return None
            
            # Generate filename with metadata
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(detection.frame_timestamp))
            filename = f"{timestamp_str}_{detection.vehicle_type.value}_{detection.confidence:.2f}_{detection.detection_id[:8]}.jpg"
            file_path = self.output_dir / filename
            
            # Save image
            success = cv2.imwrite(str(file_path), vehicle_crop)
            if success:
                self.saved_count += 1
                logger.debug(f"Saved candidate image: {filename}")
                return file_path
            else:
                logger.error(f"Failed to save candidate image: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving candidate for detection {detection.detection_id}: {e}")
            return None


class TrafficMetryProcessor:
    """Main TrafficMetry application processor."""
    
    def __init__(self, config: Config) -> None:
        """Initialize TrafficMetry processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.running = False
        self.frame_count = 0
        self.detection_count = 0
        self.event_count = 0
        
        # Initialize components
        logger.info("Initializing TrafficMetry components...")
        
        try:
            self.camera = CameraStream(config.camera)
            self.detector = VehicleDetector(config.model)
            self.lane_analyzer = LaneAnalyzer(config.lanes)
            self.event_generator = EventGenerator()
            self.candidate_saver = CandidateSaver()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self) -> None:
        """Main processing loop."""
        self._setup_signal_handlers()
        self.running = True
        
        logger.info("Starting TrafficMetry main processing loop...")
        
        # Performance monitoring
        start_time = time.time()
        last_stats_time = start_time
        
        try:
            with self.camera:
                logger.info("Camera connection established")
                
                while self.running:
                    try:
                        # Capture frame
                        frame = self.camera.get_frame()
                        if frame is None:
                            logger.warning("No frame received from camera")
                            await asyncio.sleep(0.1)
                            continue
                        
                        self.frame_count += 1
                        
                        # Detect vehicles
                        detections = self.detector.detect_vehicles(frame)
                        self.detection_count += len(detections)
                        
                        # Process each detection
                        for detection in detections:
                            await self._process_detection(frame, detection)
                        
                        # Log statistics periodically
                        current_time = time.time()
                        if current_time - last_stats_time >= 60:  # Every minute
                            self._log_statistics(current_time - start_time)
                            last_stats_time = current_time
                        
                        # Small delay to prevent CPU overload
                        await asyncio.sleep(0.01)
                        
                    except DetectionError as e:
                        logger.error(f"Detection error: {e}")
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Unexpected error in processing loop: {e}")
                        await asyncio.sleep(1)
        
        except CameraConnectionError as e:
            logger.error(f"Camera connection failed: {e}")
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            
        finally:
            logger.info("TrafficMetry processing stopped")
            self._log_final_statistics()
    
    async def _process_detection(self, frame: NDArray, detection: DetectionResult) -> None:
        """Process a single vehicle detection.
        
        Args:
            frame: Source video frame
            detection: Vehicle detection result
        """
        try:
            # Assign lane and direction
            lane_number, direction = self.lane_analyzer.assign_lane(detection)
            
            # Generate API event
            event = self.event_generator.create_vehicle_event(detection, lane_number, direction)
            self.event_count += 1
            
            # Log event (in future this will be sent via WebSocket)
            logger.info(f"Vehicle event: {event['vehicleType']} in lane {lane_number} moving {direction}")
            
            # Save candidate image (with some probability to avoid too many images)
            if detection.confidence > 0.7 and self.frame_count % 10 == 0:  # Save every 10th high-confidence detection
                await asyncio.get_event_loop().run_in_executor(
                    None, self.candidate_saver.save_candidate, frame, detection
                )
            
        except Exception as e:
            logger.error(f"Error processing detection {detection.detection_id}: {e}")
    
    def _log_statistics(self, elapsed_time: float) -> None:
        """Log performance statistics.
        
        Args:
            elapsed_time: Elapsed time since start in seconds
        """
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        detections_per_minute = (self.detection_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        logger.info(
            f"Statistics - Frames: {self.frame_count}, "
            f"Detections: {self.detection_count}, "
            f"Events: {self.event_count}, "
            f"FPS: {fps:.1f}, "
            f"Det/min: {detections_per_minute:.1f}, "
            f"Candidates saved: {self.candidate_saver.saved_count}"
        )
    
    def _log_final_statistics(self) -> None:
        """Log final statistics on shutdown."""
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total detections: {self.detection_count}")
        logger.info(f"Total events generated: {self.event_count}")
        logger.info(f"Candidate images saved: {self.candidate_saver.saved_count}")
        
        detector_info = self.detector.get_model_info()
        logger.info(f"Detector info: {detector_info}")
    
    def stop(self) -> None:
        """Stop the processing loop."""
        self.running = False
        logger.info("Shutdown requested")


async def main() -> None:
    """Main entry point."""
    logger.info("TrafficMetry starting up...")
    
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Configuration loaded - Camera: {config.camera.url}, Model: {config.model.path}")
        
        # Create and run processor
        processor = TrafficMetryProcessor(config)
        await processor.run()
        
    except ModelLoadError as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Make sure YOLO model file exists and ultralytics is installed")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal startup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        sys.exit(1)