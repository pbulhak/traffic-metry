"""TrafficMetry Main Application - Vehicle Detection and Analysis System.

This module contains the main TrafficMetryProcessor class that orchestrates
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
from typing import Any

from numpy.typing import NDArray

from backend.async_components import AsyncCameraStream, AsyncVehicleDetector
from backend.camera_stream import CameraConnectionError
from backend.candidate_saver import EventDrivenCandidateSaver
from backend.config import Settings, get_config
from backend.database import DatabaseError, EventDatabase
from backend.detector import DetectionError, ModelLoadError
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

        # Force reconnect flag (set after model loading to clear camera buffer)
        self._model_just_loaded = False

        # Performance tracking for headless monitoring
        self.camera_fps_tracker: dict[str, Any] = {
            "last_frame_time": 0.0,
            "frame_intervals": [],
            "window_size": 30,  # Rolling average over 30 frames
        }

        self.processing_fps_tracker: dict[str, Any] = {
            "processing_times": [],
            "window_size": 30,  # Rolling average over 30 processing cycles
        }

        # Initialize components
        logger.info("Initializing TrafficMetry components...")

        try:
            # Initialize async components with thread pools
            self.camera = AsyncCameraStream(config.camera, max_workers=2)
            self.detector = AsyncVehicleDetector(config.model, max_workers=3)

            # Initialize database first to get journey continuation
            self.event_database = EventDatabase(config.database)

            # We'll initialize the tracking manager after database connection
            self.vehicle_tracking_manager: VehicleTrackingManager | None = None

            # Event-driven candidate saver (replaces old CandidateSaver)
            self.event_candidate_saver = EventDrivenCandidateSaver(
                output_dir=Path("data/unlabeled_images"),
                storage_limit_gb=config.model.candidate_storage_limit_gb,
            )

            # Validate ROI configuration
            if config.roi.enabled:
                if not config.roi.validate_coordinates(config.camera.width, config.camera.height):
                    raise ValueError(
                        f"Invalid ROI coordinates: ({config.roi.x1},{config.roi.y1}) "
                        f"to ({config.roi.x2},{config.roi.y2}) "
                        f"for frame size {config.camera.width}x{config.camera.height}"
                    )
                roi_dims = config.roi.get_roi_dimensions()
                logger.info(
                    f"ROI enabled: ({config.roi.x1},{config.roi.y1}) "
                    f"to ({config.roi.x2},{config.roi.y2}) "
                    f"[{roi_dims[0]}x{roi_dims[1]} px]"
                )
            else:
                logger.info("ROI disabled - processing full frame")

            logger.info("Basic components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def _initialize_tracking_manager(self) -> None:
        """Initialize vehicle tracking manager with journey ID continuation."""
        try:
            # Get last journey ID from database for continuation
            last_journey_id = await self.event_database.get_last_journey_id()

            # Initialize tracking manager with all parameters from config
            self.vehicle_tracking_manager = VehicleTrackingManager(
                config=self.config.model,  # ALL ByteTrack parameters from config
                frame_rate=self.config.camera.fps,  # Camera frame rate from config
                update_interval_seconds=self.config.server.tracker_update_interval_seconds,  # Update interval from config
                start_journey_counter=last_journey_id,  # Continue journey IDs from database
            )

            # Register event-driven candidate saver as listener
            self.vehicle_tracking_manager.add_event_listener(self._handle_tracking_event)

            logger.info(
                f"Vehicle tracking manager initialized with journey ID continuation from {last_journey_id}"
            )
            logger.info("Event-driven candidate saver registered as listener")

            # Set flag to trigger camera reconnect
            # (model was loaded in detector.initialize() before this method)
            self._model_just_loaded = True
            logger.info("Model loaded - will force camera reconnect on next frame")

        except Exception as e:
            logger.error(f"Failed to initialize vehicle tracking manager: {e}")
            raise

    def _extract_roi(self, frame: NDArray) -> NDArray:
        """Extract Region of Interest from frame if ROI is enabled.

        Args:
            frame: Full frame from camera

        Returns:
            ROI cropped frame if ROI enabled, otherwise full frame
        """
        if not self.config.roi.enabled:
            return frame

        # Extract ROI using numpy slicing (very efficient)
        roi_frame = frame[
            self.config.roi.y1 : self.config.roi.y2, self.config.roi.x1 : self.config.roi.x2
        ]

        return roi_frame

    def _offset_detections_from_roi(
        self, detections: list[DetectionResult]
    ) -> list[DetectionResult]:
        """Offset detection coordinates from ROI back to full frame coordinates.

        Args:
            detections: List of detections in ROI coordinate system

        Returns:
            List of detections in full frame coordinate system
        """
        if not self.config.roi.enabled:
            return detections

        offset_x = self.config.roi.x1
        offset_y = self.config.roi.y1

        offset_detections = []

        for det in detections:
            # Create new DetectionResult with offset coordinates
            offset_det = DetectionResult(
                detection_id=det.detection_id,
                vehicle_type=det.vehicle_type,
                confidence=det.confidence,
                class_id=det.class_id,
                x1=det.x1 + offset_x,
                y1=det.y1 + offset_y,
                x2=det.x2 + offset_x,
                y2=det.y2 + offset_y,
                frame_timestamp=det.frame_timestamp,
                frame_id=det.frame_id,
                frame_shape=det.frame_shape,  # Keep original frame shape
                track_id=det.track_id,
            )
            offset_detections.append(offset_det)

        return offset_detections

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

            # Initialize async detector
            await self.detector.initialize()
            logger.info("Async vehicle detector initialized")

            # Initialize tracking manager with journey ID continuation
            await self._initialize_tracking_manager()
            logger.info("Vehicle tracking manager initialized with journey ID continuation")

            # Connect to async camera stream
            async with self.camera:
                logger.info("Async camera stream connected")

                while self.running:
                    try:
                        # 📊 PERFORMANCE TRACKING: Start processing cycle
                        processing_start = time.time()

                        # 📊 CAMERA FPS TRACKING: Measure frame interval
                        camera_frame_start = time.time()
                        frame = await self.camera.get_frame_with_reconnect()

                        # ⚡ FORCE RECONNECT: Clear camera buffer after model loading
                        if self._model_just_loaded:
                            logger.info("🔄 Model just loaded, forcing camera reconnect to clear buffer...")
                            success = await self.camera.force_reconnect()
                            if success:
                                logger.info("✅ Camera reconnected successfully after model load")
                                # Get fresh frame after reconnect
                                frame = await self.camera.get_frame_with_reconnect()
                            else:
                                logger.error("❌ Failed to force camera reconnect")
                            self._model_just_loaded = False  # Execute only once

                        # Track camera FPS (frame delivery rate)
                        if self.camera_fps_tracker["last_frame_time"] > 0:
                            frame_interval = (
                                camera_frame_start - self.camera_fps_tracker["last_frame_time"]
                            )
                            self.camera_fps_tracker["frame_intervals"].append(frame_interval)

                            # Maintain rolling window
                            if (
                                len(self.camera_fps_tracker["frame_intervals"])
                                > self.camera_fps_tracker["window_size"]
                            ):
                                self.camera_fps_tracker["frame_intervals"].pop(0)

                        self.camera_fps_tracker["last_frame_time"] = camera_frame_start

                        if frame is None:
                            logger.warning("No frame received from camera")
                            await asyncio.sleep(0.1)
                            continue

                        self.frame_count += 1

                        # 🎨 ROI EXTRACTION: Extract region of interest if enabled
                        roi_frame = self._extract_roi(frame)

                        # Detect vehicles asynchronously in ROI (raw detections from YOLO)
                        raw_detections_roi = await self.detector.detect_vehicles(roi_frame)

                        # 📐 ROI OFFSET: Transform coordinates back to full frame
                        raw_detections = self._offset_detections_from_roi(raw_detections_roi)
                        self.detection_count += len(raw_detections)

                        # 🎯 EVENT-DRIVEN TRACKING: Update tracking with dynamic direction detection
                        assert self.vehicle_tracking_manager is not None, (
                            "Tracking manager should be initialized"
                        )
                        tracked_detections, vehicle_events = self.vehicle_tracking_manager.update(
                            raw_detections, current_frame=frame
                        )

                        # Process vehicle lifecycle events (not detections!)
                        await self._process_vehicle_events(vehicle_events)

                        # 📊 PROCESSING FPS TRACKING: Complete cycle timing
                        processing_end = time.time()
                        processing_time = processing_end - processing_start
                        self.processing_fps_tracker["processing_times"].append(processing_time)

                        # Maintain rolling window for processing times
                        if (
                            len(self.processing_fps_tracker["processing_times"])
                            > self.processing_fps_tracker["window_size"]
                        ):
                            self.processing_fps_tracker["processing_times"].pop(0)

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
                    f"🚗 Vehicle {event.journey_id} ({event.vehicle_type}) entered at "
                    f"position ({event.detection.centroid[0]}, {event.detection.centroid[1]})"
                )

                # Future: Send WebSocket notification for vehicle entry
                # await self._publish_websocket_event(event.to_websocket_format())

            elif isinstance(event, VehicleUpdated):
                # Vehicle position updated with dynamic direction analysis
                logger.debug(
                    f"📍 Vehicle {event.journey_id} updated: "
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
                            f"🏁 Vehicle {journey.journey_id} journey completed and saved: "
                            f"{journey.journey_duration_seconds:.1f}s, "
                            f"{journey.total_detections} detections, "
                            f"best confidence {journey.best_confidence:.2f}"
                        )
                    else:
                        logger.error(f"Failed to save journey for vehicle {event.journey_id}")

                except DatabaseError as e:
                    logger.error(f"Database error saving vehicle {event.journey_id} journey: {e}")

                # Future: Send WebSocket exit notification
                # await self._publish_websocket_event(event.to_websocket_format())

    def _log_statistics(self, elapsed_time: float) -> None:
        """Log comprehensive performance statistics with headless monitoring metrics.

        Args:
            elapsed_time: Elapsed time since start in seconds
        """
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        detections_per_minute = (
            (self.detection_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        )

        # 📊 CALCULATE CAMERA FPS (frame delivery rate from AsyncCameraStream)
        camera_fps = 0.0
        if len(self.camera_fps_tracker["frame_intervals"]) > 1:
            avg_frame_interval = sum(self.camera_fps_tracker["frame_intervals"]) / len(
                self.camera_fps_tracker["frame_intervals"]
            )
            camera_fps = 1.0 / avg_frame_interval if avg_frame_interval > 0 else 0.0

        # 📊 CALCULATE PROCESSING FPS (complete processing cycle rate)
        processing_fps = 0.0
        if len(self.processing_fps_tracker["processing_times"]) > 1:
            avg_processing_time = sum(self.processing_fps_tracker["processing_times"]) / len(
                self.processing_fps_tracker["processing_times"]
            )
            processing_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0

        # 📊 CALCULATE EFFICIENCY RATIO (how well we utilize camera stream)
        efficiency_ratio = 0.0
        if camera_fps > 0:
            efficiency_ratio = (processing_fps / camera_fps) * 100

        assert self.vehicle_tracking_manager is not None
        tracking_stats = self.vehicle_tracking_manager.get_tracking_stats()

        # 🚀 ENHANCED PERFORMANCE LOGGING with headless monitoring
        logger.info(
            f"🎯 PERFORMANCE METRICS | "
            f"Camera FPS: {camera_fps:.1f}, "
            f"Processing FPS: {processing_fps:.1f}, "
            f"Pipeline Efficiency: {efficiency_ratio:.1f}% | "
            f"Overall FPS: {fps:.1f}, "
            f"Frames: {self.frame_count}, "
            f"Detections: {self.detection_count} ({detections_per_minute:.1f}/min) | "
            f"🚗 TRACKING | "
            f"Active: {tracking_stats['active_vehicles']}, "
            f"Total: {tracking_stats['total_vehicles_tracked']}, "
            f"Journeys: {tracking_stats['total_journeys_completed']}, "
            f"Events: {self.event_count} | "
            f"💾 STORAGE | "
            f"Candidates: {self.event_candidate_saver.get_statistics()['saved_candidates']}"
        )

        # 📊 ADDITIONAL DETAILED METRICS (every 5 minutes for deep analysis)
        if elapsed_time > 0 and int(elapsed_time) % 300 == 0:  # Every 5 minutes
            frame_intervals_count = len(self.camera_fps_tracker["frame_intervals"])
            processing_times_count = len(self.processing_fps_tracker["processing_times"])

            logger.info(
                f"📊 DEEP METRICS (5min) | "
                f"Camera samples: {frame_intervals_count}/{self.camera_fps_tracker['window_size']}, "
                f"Processing samples: {processing_times_count}/{self.processing_fps_tracker['window_size']} | "
                f"Avg frame interval: {avg_frame_interval * 1000:.1f}ms, "
                f"Avg processing time: {avg_processing_time * 1000:.1f}ms | "
                f"Thread pools: Camera={self.camera.get_camera_info().get('worker_threads', 'N/A')}, "
                f"Detector={self.detector.get_model_info().get('async_worker_threads', 'N/A')}"
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
