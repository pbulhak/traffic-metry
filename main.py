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

from numpy.typing import NDArray

from backend.camera_stream import CameraConnectionError, CameraStream
from backend.candidate_saver import EventDrivenCandidateSaver
from backend.config import Settings, get_config
from backend.database import DatabaseError, EventDatabase
from backend.detector import DetectionError, ModelLoadError, VehicleDetector
from backend.tracker import VehicleTrackingManager
from backend.vehicle_events import VehicleEntered, VehicleEvent, VehicleExited, VehicleUpdated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trafficmetry.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# LaneAnalyzer class removed - replaced with dynamic direction detection
# See backend/direction_analyzer.py for the new approach


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

            # Initialize database first to get journey continuation
            self.event_database = EventDatabase(config.database)

            # We'll initialize the tracking manager after database connection
            self.vehicle_tracking_manager: VehicleTrackingManager | None = None

            # Event-driven candidate saver (replaces old CandidateSaver)
            self.event_candidate_saver = EventDrivenCandidateSaver(
                output_dir=Path("data/unlabeled_images"),
                storage_limit_gb=config.model.candidate_storage_limit_gb,
            )

            logger.info("Basic components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def _initialize_tracking_manager(self) -> None:
        """Initialize vehicle tracking manager with journey ID continuation."""
        try:
            # Get last journey ID from database for continuation
            last_journey_id = await self.event_database.get_last_journey_id()

            # Initialize tracking manager with track confirmation
            self.vehicle_tracking_manager = VehicleTrackingManager(
                track_activation_threshold=0.5,  # Detection confidence threshold for track activation
                lost_track_buffer=30,  # Number of frames to buffer when a track is lost
                minimum_matching_threshold=0.8,  # Threshold for matching tracks with detections
                frame_rate=30,  # Video frame rate for prediction algorithms
                update_interval_seconds=1.0,  # WebSocket update interval
                start_journey_counter=last_journey_id,  # Continue journey IDs from database
                minimum_consecutive_frames=3,  # Track confirmation threshold
            )

            # Register event-driven candidate saver as listener
            self.vehicle_tracking_manager.add_event_listener(self._handle_tracking_event)

            logger.info(
                f"Vehicle tracking manager initialized with journey ID continuation from {last_journey_id}"
            )
            logger.info("Event-driven candidate saver registered as listener")

        except Exception as e:
            logger.error(f"Failed to initialize vehicle tracking manager: {e}")
            raise

    def _handle_tracking_event(self, event: VehicleEvent, frame: NDArray) -> None:
        """Handle vehicle tracking events by routing them to the candidate saver.

        Args:
            event: Vehicle lifecycle event
            frame: Current video frame
        """
        try:
            if isinstance(event, VehicleEntered):
                self.event_candidate_saver.handle_vehicle_entered(event, frame)
            elif isinstance(event, VehicleUpdated):
                self.event_candidate_saver.handle_vehicle_updated(event, frame)
            elif isinstance(event, VehicleExited):
                saved_path = self.event_candidate_saver.handle_vehicle_exited(event)
                if saved_path:
                    logger.debug(f"Candidate image saved for {event.journey_id}: {saved_path.name}")
        except Exception as e:
            logger.error(f"Error handling tracking event {type(event).__name__}: {e}")

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

            # Initialize tracking manager with journey ID continuation
            await self._initialize_tracking_manager()
            logger.info("Vehicle tracking manager initialized with journey ID continuation")

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
                        assert self.vehicle_tracking_manager is not None, "Tracking manager should be initialized"
                        tracked_detections, vehicle_events = self.vehicle_tracking_manager.update(
                            raw_detections, current_frame=frame
                        )

                        # Process vehicle lifecycle events (not detections!)
                        await self._process_vehicle_events(vehicle_events)

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
                    f"direction {event.movement_direction}, "
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

    def _log_statistics(self, elapsed_time: float) -> None:
        """Log performance statistics.

        Args:
            elapsed_time: Elapsed time since start in seconds
        """
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        detections_per_minute = (
            (self.detection_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        )

        assert self.vehicle_tracking_manager is not None
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
            f"Candidates saved: {self.event_candidate_saver.get_statistics()['saved_candidates']}"
        )

    def _log_final_statistics(self) -> None:
        """Log final statistics on shutdown."""
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total detections: {self.detection_count}")
        logger.info(f"Total events generated: {self.event_count}")
        candidate_stats = self.event_candidate_saver.get_statistics()
        logger.info(f"Candidate images saved: {candidate_stats['saved_candidates']}")

        assert self.vehicle_tracking_manager is not None
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
