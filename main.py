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
from pathlib import Path

import cv2
from numpy.typing import NDArray

from backend.camera_stream import CameraConnectionError, CameraStream
from backend.config import ModelSettings, Settings, get_config
from backend.database import DatabaseError, EventDatabase
from backend.detection_models import DetectionResult
from backend.detector import DetectionError, ModelLoadError, VehicleDetector
from backend.tracker import VehicleTrackingManager
from backend.vehicle_events import VehicleEntered, VehicleExited, VehicleUpdated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trafficmetry.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# LaneAnalyzer class removed - replaced with dynamic direction detection
# See backend/direction_analyzer.py for the new approach



class CandidateSaver:
    """Saves detected vehicle images as training candidates."""

    def __init__(self, output_dir: Path, model_config: ModelSettings) -> None:
        """Initialize candidate saver.

        Args:
            output_dir: Directory to save candidate images
            model_config: Model configuration with storage settings
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0

        # Storage management from configuration
        self.storage_limit_bytes = int(model_config.candidate_storage_limit_gb * 1024**3)
        self.cleanup_interval = model_config.candidate_cleanup_interval
        self.cleanup_counter = 0

        logger.info(f"Candidate saver initialized - output directory: {self.output_dir}")
        logger.info(
            f"Storage limit: {model_config.candidate_storage_limit_gb:.1f} GB ({self.storage_limit_bytes:,} bytes)"
        )
        logger.info(f"Cleanup interval: every {self.cleanup_interval} saves")

    def _get_directory_size_bytes(self) -> int:
        """Calculate total size of all files in output directory.

        Returns:
            Total size in bytes, or 0 if directory doesn't exist or is empty
        """
        try:
            if not self.output_dir.exists():
                return 0

            total_size = 0
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            return total_size
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
            return 0

    def _check_and_cleanup_storage(self) -> None:
        """Check storage usage and cleanup oldest files if limit exceeded."""
        try:
            current_size = self._get_directory_size_bytes()
            if current_size <= self.storage_limit_bytes:
                logger.debug(
                    f"Storage within limit: {current_size:,} / {self.storage_limit_bytes:,} bytes"
                )
                return

            logger.info(
                f"Storage limit exceeded: {current_size:,} / {self.storage_limit_bytes:,} bytes - starting cleanup"
            )

            # Get all image files sorted by modification time (oldest first)
            image_files = []
            for file_path in self.output_dir.glob("*.jpg"):
                if file_path.is_file():
                    image_files.append((file_path, file_path.stat().st_mtime))

            # Sort by modification time (oldest first)
            image_files.sort(key=lambda x: x[1])

            # Delete oldest files until we're under limit (with 10% buffer)
            target_size = int(self.storage_limit_bytes * 0.9)  # Keep 10% buffer
            deleted_count = 0
            freed_bytes = 0

            for file_path, _ in image_files:
                if current_size - freed_bytes <= target_size:
                    break

                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()  # Delete file
                    freed_bytes += file_size
                    deleted_count += 1
                    logger.debug(f"Deleted old candidate: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

            final_size = current_size - freed_bytes
            logger.info(
                f"Cleanup completed: deleted {deleted_count} files, freed {freed_bytes:,} bytes"
            )
            logger.info(f"Final storage: {final_size:,} / {self.storage_limit_bytes:,} bytes")

        except Exception as e:
            logger.error(f"Error during storage cleanup: {e}")

    def save_candidate(self, frame: NDArray, detection: DetectionResult) -> Path | None:
        """Save vehicle detection as candidate image.

        Args:
            frame: Source video frame
            detection: Vehicle detection result

        Returns:
            Path to saved image or None if saving failed
        """
        try:
            # BOUNDS VALIDATION - ensure coordinates are within frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Clamp coordinates to frame boundaries
            x1 = max(0, min(detection.x1, frame_width - 1))
            y1 = max(0, min(detection.y1, frame_height - 1))
            x2 = max(x1 + 1, min(detection.x2, frame_width))  # Ensure x2 > x1
            y2 = max(y1 + 1, min(detection.y2, frame_height))  # Ensure y2 > y1

            # Log if bounds were adjusted
            if (x1, y1, x2, y2) != (detection.x1, detection.y1, detection.x2, detection.y2):
                logger.debug(
                    f"Adjusted bounds for detection {detection.detection_id}: "
                    f"({detection.x1},{detection.y1},{detection.x2},{detection.y2}) -> ({x1},{y1},{x2},{y2})"
                )

            # SAFE CROPPING with validated coordinates
            vehicle_crop = frame[y1:y2, x1:x2]

            if vehicle_crop.size == 0:
                logger.warning(
                    f"Empty crop for detection {detection.detection_id} after bounds validation"
                )
                return None

            # Generate filename with metadata
            timestamp_str = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(detection.frame_timestamp)
            )
            filename = f"{timestamp_str}_{detection.vehicle_type.value}_{detection.confidence:.2f}_{detection.detection_id[:8]}.jpg"
            file_path = self.output_dir / filename

            # Save image
            success = cv2.imwrite(str(file_path), vehicle_crop)
            if success:
                self.saved_count += 1
                logger.debug(f"Saved candidate image: {filename}")

                # CLEANUP TRIGGER - check storage periodically
                self.cleanup_counter += 1
                if self.cleanup_counter >= self.cleanup_interval:
                    self.cleanup_counter = 0
                    self._check_and_cleanup_storage()

                return file_path
            else:
                logger.error(f"Failed to save candidate image: {filename}")
                return None

        except Exception as e:
            logger.error(f"Error saving candidate for detection {detection.detection_id}: {e}")
            return None


class TrafficMetryProcessor:
    """Main TrafficMetry application processor."""

    def __init__(self, config: Settings) -> None:
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
            self.vehicle_tracking_manager = VehicleTrackingManager(
                track_activation_threshold=0.5,    # Detection confidence threshold for track activation
                lost_track_buffer=30,               # Number of frames to buffer when a track is lost
                minimum_matching_threshold=0.8,    # Threshold for matching tracks with detections
                frame_rate=30,                     # Video frame rate for prediction algorithms
                update_interval_seconds=1.0        # WebSocket update interval
            )
            # LaneAnalyzer removed - dynamic direction detection now handled in VehicleTrackingManager
            self.candidate_saver = CandidateSaver(Path("data/unlabeled_images"), config.model)
            self.event_database = EventDatabase(config.database)

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: object) -> None:
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
            # Connect to database first
            await self.event_database.connect()
            logger.info("Database connection established")

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

                        # Detect vehicles (raw detections from YOLO)
                        raw_detections = self.detector.detect_vehicles(frame)
                        self.detection_count += len(raw_detections)

                        # 🎯 EVENT-DRIVEN TRACKING: Update tracking with dynamic direction detection
                        tracked_detections, vehicle_events = self.vehicle_tracking_manager.update(
                            raw_detections
                        )

                        # Process vehicle lifecycle events (not detections!)
                        await self._process_vehicle_events(vehicle_events)

                        # Save candidate images for active vehicles
                        for detection in tracked_detections:
                            await self._save_candidate_image(frame, detection)

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
            # Close database connection
            await self.event_database.close()
            logger.info("TrafficMetry processing stopped")
            self._log_final_statistics()

    async def _process_vehicle_events(self, vehicle_events: list) -> None:
        """Process vehicle lifecycle events from VehicleTrackingManager.

        Args:
            vehicle_events: List of VehicleEvent objects (VehicleEntered, VehicleUpdated, VehicleExited)
        """
        for event in vehicle_events:
            if isinstance(event, VehicleEntered):
                # Vehicle entered tracking area with dynamic direction detection
                logger.info(
                    f"🚗 Vehicle {event.journey_id} (Track {event.track_id}) ({event.vehicle_type}) entered at "
                    f"position ({event.detection.centroid[0]}, {event.detection.centroid[1]})"
                )

                # Future: Send WebSocket notification for vehicle entry
                # await self._publish_websocket_event(event.to_websocket_format())

            elif isinstance(event, VehicleUpdated):
                # Vehicle position updated with dynamic direction analysis
                logger.debug(
                    f"📍 Vehicle {event.journey_id} (Track {event.track_id}) updated: "
                    f"direction {event.movement_direction} ({event.direction_confidence:.2f}), "
                    f"confidence {event.current_confidence:.2f}, "
                    f"detections {event.total_detections_so_far}"
                )

                # Future: Send real-time WebSocket update
                # await self._publish_websocket_event(event.to_websocket_format())

            elif isinstance(event, VehicleExited):
                # Vehicle exited - save complete journey to database
                journey = event.journey

                try:
                    success = await self.event_database.save_vehicle_journey(journey)
                    if success:
                        self.event_count += 1
                        logger.info(
                            f"🏁 Vehicle {event.track_id} journey completed and saved: "
                            f"{journey.journey_duration_seconds:.1f}s, "
                            f"{journey.total_detections} detections, "
                            f"best confidence {journey.best_confidence:.2f}"
                        )
                    else:
                        logger.error(f"Failed to save journey for vehicle {event.track_id}")

                except DatabaseError as e:
                    logger.error(f"Database error saving vehicle {event.track_id} journey: {e}")

                # Future: Send WebSocket exit notification
                # await self._publish_websocket_event(event.to_websocket_format())

    async def _save_candidate_image(self, frame: NDArray, detection: DetectionResult) -> None:
        """Save candidate image for tracked vehicle.

        Args:
            frame: Source video frame
            detection: Vehicle detection result with track_id
        """
        try:
            # Save candidate image (with some probability to avoid too many images)
            if (
                detection.confidence > 0.7 and self.frame_count % 10 == 0
            ):  # Save every 10th high-confidence detection
                await asyncio.get_event_loop().run_in_executor(
                    None, self.candidate_saver.save_candidate, frame, detection
                )

        except Exception as e:
            logger.error(f"Error saving candidate image for detection {detection.detection_id}: {e}")

    def _log_statistics(self, elapsed_time: float) -> None:
        """Log performance statistics.

        Args:
            elapsed_time: Elapsed time since start in seconds
        """
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        detections_per_minute = (
            (self.detection_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        )

        tracking_stats = self.vehicle_tracking_manager.get_tracking_stats()

        logger.info(
            f"Statistics - Frames: {self.frame_count}, "
            f"Detections: {self.detection_count}, "
            f"Events: {self.event_count}, "
            f"FPS: {fps:.1f}, "
            f"Det/min: {detections_per_minute:.1f}, "
            f"Active vehicles: {tracking_stats['active_vehicles']}, "
            f"Total vehicles: {tracking_stats['total_vehicles_tracked']}, "
            f"Journeys completed: {tracking_stats['total_journeys_completed']}, "
            f"Candidates saved: {self.candidate_saver.saved_count}"
        )

    def _log_final_statistics(self) -> None:
        """Log final statistics on shutdown."""
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total detections: {self.detection_count}")
        logger.info(f"Total events generated: {self.event_count}")
        logger.info(f"Candidate images saved: {self.candidate_saver.saved_count}")

        tracking_stats = self.vehicle_tracking_manager.get_tracking_stats()
        logger.info(f"Total vehicles tracked: {tracking_stats['total_vehicles_tracked']}")
        logger.info(f"Complete journeys recorded: {tracking_stats['total_journeys_completed']}")

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
        logger.info(
            f"Configuration loaded - Camera: {config.camera.url}, Model: {config.model.path}"
        )

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
