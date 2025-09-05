"""Threading logic for diagnostics viewer.

This module contains the ProcessingThread and simplified GUIThread classes
that handle the asynchronous processing pipeline and GUI interaction loop.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.async_components import AsyncCameraStream, AsyncVehicleDetector
from backend.candidate_saver import EventDrivenCandidateSaver
from backend.config import Settings
from backend.database import EventDatabase
from backend.tracker import VehicleTrackingManager
from backend.vehicle_events import VehicleEntered, VehicleExited, VehicleUpdated

from .renderer import DiagnosticsRenderer
from .state import ShutdownCoordinator, ThreadSafeDataManager
from .utils import DEFAULT_WINDOW_CONFIG, FPS_TRACKING_CONFIG, display_unified_help

logger = logging.getLogger(__name__)


class ProcessingThread:
    """Asynchronous processing thread for camera, detection, and tracking."""

    def __init__(
        self,
        config: Settings,
        data_manager: ThreadSafeDataManager,
        shutdown_coordinator: ShutdownCoordinator,
    ):
        """Initialize processing thread components."""
        self.config = config
        self.data_manager = data_manager
        self.shutdown_coordinator = shutdown_coordinator

        # Initialize async components
        self.camera = AsyncCameraStream(config.camera, max_workers=2)
        self.detector = AsyncVehicleDetector(config.model, max_workers=3)

        # Initialize tracking and database components
        self.event_database = EventDatabase(config.database)
        self.vehicle_tracking_manager: VehicleTrackingManager | None = None
        self.event_candidate_saver = EventDrivenCandidateSaver(
            output_dir=Path("data/unlabeled_images"),
            storage_limit_gb=config.model.candidate_storage_limit_gb,
        )

        # Session tracking
        self.session_duration: float = 0.0
        self._last_bottleneck_log_time = 0.0

        # Processing statistics
        self.frame_count = 0
        self.detection_count = 0

        # Performance debugging metrics
        self._processing_bottleneck_tracker: dict[str, Any] = {
            "camera_wait_times": [],
            "detection_times": [],
            "tracking_times": [],
            "data_update_times": [],
            "window_size": 10,  # Smaller window for bottleneck analysis
        }
        self.event_count = 0

        # Processing FPS tracking
        self.processing_fps_tracker: dict[str, Any] = {
            "processing_times": [],
            "window_size": 30,  # Rolling average over 30 frames
            "last_processing_start": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize async components."""
        try:
            # Connect database
            await self.event_database.connect()
            logger.info("Database connected in processing thread")

            # Initialize async detector
            await self.detector.initialize()
            logger.info("Async detector initialized in processing thread")

            # Initialize tracking manager with configurable parameters
            last_journey_id = await self.event_database.get_last_journey_id()
            self.vehicle_tracking_manager = VehicleTrackingManager(
                frame_rate=30,
                update_interval_seconds=1.0,
                start_journey_counter=last_journey_id,
                minimum_consecutive_frames=5,
            )
            # Register event listener
            self.vehicle_tracking_manager.add_event_listener(self._handle_tracking_event)
            logger.info("Vehicle tracking manager initialized in processing thread")

        except Exception as e:
            logger.error(f"Failed to initialize processing thread: {e}")
            raise

    def _handle_tracking_event(self, event: Any, frame: NDArray) -> None:
        """Handle tracking events for candidate saving."""
        try:
            if isinstance(event, VehicleEntered):
                self.event_candidate_saver.handle_vehicle_entered(event, frame)
            elif isinstance(event, VehicleUpdated):
                self.event_candidate_saver.handle_vehicle_updated(event, frame)
            elif isinstance(event, VehicleExited):
                saved_path = self.event_candidate_saver.handle_vehicle_exited(event)
                if saved_path:
                    logger.debug(f"Candidate saved: {saved_path.name}")
        except Exception as e:
            logger.error(f"Error handling tracking event: {e}")

    async def processing_loop(self) -> None:
        """Main asynchronous processing loop."""
        logger.info("Starting processing thread loop")

        # Track session start time
        session_start_time = time.time()

        try:
            await self.initialize()

            async with self.camera:
                logger.info("Processing thread: Camera connected")

                while not self.shutdown_coordinator.is_shutdown_requested():
                    try:
                        # 🎮 CHECK CONTROL STATE
                        control_state = self.data_manager.get_control_state()

                        if control_state.processing_paused:
                            await asyncio.sleep(0.1)
                            continue

                        # 📊 PROCESSING FPS TRACKING: Start cycle
                        processing_start = time.time()

                        # 🎯 ASYNC PROCESSING PIPELINE with bottleneck tracking
                        camera_start = time.time()
                        frame = await self.camera.get_frame()
                        camera_time = time.time() - camera_start
                        self._track_bottleneck_metric("camera_wait_times", camera_time)

                        if frame is None:
                            await asyncio.sleep(0.1)
                            continue

                        self.frame_count += 1

                        # Conditional detection based on control state
                        detection_start = time.time()
                        if control_state.detection_enabled:
                            raw_detections = await self.detector.detect_vehicles(frame)
                            detection_time = time.time() - detection_start
                            self._track_bottleneck_metric("detection_times", detection_time)
                            self.detection_count += len(raw_detections)
                        else:
                            raw_detections = []  # Skip detection entirely
                            detection_time = 0.0
                            self._track_bottleneck_metric("detection_times", detection_time)

                        # Conditional tracking based on control state
                        tracking_start = time.time()
                        if control_state.tracking_enabled:
                            assert self.vehicle_tracking_manager is not None
                            # Always call update (even with empty detections) to allow finalization of lost tracks
                            tracked_detections, vehicle_events = self.vehicle_tracking_manager.update(
                                raw_detections, current_frame=frame
                            )
                            tracking_time = time.time() - tracking_start
                            self._track_bottleneck_metric("tracking_times", tracking_time)
                        else:
                            tracked_detections = []
                            vehicle_events = []
                            tracking_time = 0.0
                            self._track_bottleneck_metric("tracking_times", tracking_time)

                        # Conditional processing based on control state
                        if control_state.database_enabled:
                            await self._process_vehicle_events(vehicle_events)

                        # Handle candidate saving based on control state
                        if not control_state.candidates_enabled:
                            # Skip candidate processing for this frame
                            pass

                        # 📊 DATA UPDATE with timing
                        data_update_start = time.time()

                        # 📊 PROCESSING FPS TRACKING: Complete cycle
                        processing_end = time.time()
                        processing_time = processing_end - processing_start
                        self.processing_fps_tracker["processing_times"].append(processing_time)

                        # Maintain sliding window
                        processing_times_list = self.processing_fps_tracker["processing_times"]
                        if len(processing_times_list) > int(self.processing_fps_tracker["window_size"]):
                            processing_times_list.pop(0)

                        # Calculate current processing FPS
                        processing_fps = self._calculate_processing_fps()

                        # 🔒 THREAD-SAFE UPDATE to shared data
                        stats = {
                            "frame_count": self.frame_count,
                            "detection_count": self.detection_count,
                            "event_count": self.event_count,
                            "tracking_stats": self.vehicle_tracking_manager.get_tracking_stats() if self.vehicle_tracking_manager else {},
                        }

                        self.data_manager.update_frame_data(
                            frame=frame,
                            raw_detections=raw_detections,
                            tracked_detections=tracked_detections,
                            vehicle_events=vehicle_events,
                            stats=stats,
                            processing_fps=processing_fps,
                        )

                        # Track data update time
                        data_update_time = time.time() - data_update_start
                        self._track_bottleneck_metric("data_update_times", data_update_time)

                        # 🔍 BOTTLENECK LOGGING (every 30 seconds)
                        current_time = time.time()
                        if current_time - self._last_bottleneck_log_time >= 30:  # Every 30 seconds
                            bottleneck_analysis = self._get_bottleneck_analysis()
                            if bottleneck_analysis:
                                logger.info(
                                    f"🔍 BOTTLENECK ANALYSIS | "
                                    f"Camera: {bottleneck_analysis.get('avg_camera_wait_ms', 0):.1f}ms | "
                                    f"Detection: {bottleneck_analysis.get('avg_detection_ms', 0):.1f}ms | "
                                    f"Tracking: {bottleneck_analysis.get('avg_tracking_ms', 0):.1f}ms | "
                                    f"Data Update: {bottleneck_analysis.get('avg_data_update_ms', 0):.1f}ms | "
                                    f"Processing FPS: {processing_fps:.1f}"
                                )
                            self._last_bottleneck_log_time = current_time

                        # Small delay to prevent CPU overload
                        await asyncio.sleep(0.01)

                    except Exception as e:
                        logger.error(f"Error in processing loop: {e}")
                        await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Fatal error in processing thread: {e}")
        finally:
            # Calculate final session duration
            self.session_duration = time.time() - session_start_time

            await self._cleanup()
            self.shutdown_coordinator.signal_processing_done()
            logger.info(f"Processing thread finished after {self.session_duration:.1f} seconds")

    def _calculate_processing_fps(self) -> float:
        """Calculate current processing FPS from sliding window."""
        processing_times = self.processing_fps_tracker["processing_times"]
        if len(processing_times) < 2:
            return 0.0

        avg_processing_time = sum(processing_times) / len(processing_times)
        return 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0

    def _track_bottleneck_metric(self, metric_name: str, measurement: float) -> None:
        """Track performance bottleneck metrics for debugging."""
        if metric_name not in self._processing_bottleneck_tracker:
            return

        measurements = self._processing_bottleneck_tracker[metric_name]
        measurements.append(measurement)

        # Maintain sliding window
        if len(measurements) > int(self._processing_bottleneck_tracker["window_size"]):
            measurements.pop(0)

    def _get_bottleneck_analysis(self) -> dict[str, float]:
        """Get bottleneck analysis for debugging."""
        analysis = {}

        for metric_name, measurements in self._processing_bottleneck_tracker.items():
            if metric_name == "window_size" or not measurements:
                continue

            avg_time = sum(measurements) / len(measurements)
            analysis[f"avg_{metric_name.replace('_times', '')}_ms"] = avg_time * 1000

        return analysis

    async def _process_vehicle_events(self, events: Any) -> None:
        """Process vehicle events (database saving, etc.)."""
        for event in events:
            if isinstance(event, VehicleExited):
                try:
                    success = await self.event_database.save_vehicle_journey(event.journey)
                    if success:
                        self.event_count += 1
                        logger.info(f"Journey saved: {event.journey.journey_id}")
                except Exception as e:
                    logger.error(f"Failed to save journey: {e}")

    async def _cleanup(self) -> None:
        """Cleanup processing thread resources."""
        try:
            if self.event_database:
                await self.event_database.close()
            logger.info("Processing thread cleanup complete")
        except Exception as e:
            logger.error(f"Error during processing thread cleanup: {e}")


class GUIThread:
    """Simplified GUI thread focused on input handling and rendering coordination."""

    def __init__(
        self,
        data_manager: ThreadSafeDataManager,
        shutdown_coordinator: ShutdownCoordinator,
        renderer: DiagnosticsRenderer
    ):
        """Initialize GUI thread with renderer dependency injection."""
        self.data_manager = data_manager
        self.shutdown_coordinator = shutdown_coordinator
        self.renderer = renderer

        # Window configuration
        self.window_name: str = DEFAULT_WINDOW_CONFIG["name"]  # type: ignore[assignment]

        # GUI FPS tracking
        self.gui_fps_tracker: dict[str, Any] = {
            "frame_times": [],
            "window_size": FPS_TRACKING_CONFIG["window_size"],
            "last_frame_time": 0.0,
        }

        # Frame caching for performance
        self._last_rendered_frame_id: int | None = None

    def setup_window(self) -> None:
        """Setup OpenCV window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name,
            int(DEFAULT_WINDOW_CONFIG["width"]),  # type: ignore[call-overload]
            int(DEFAULT_WINDOW_CONFIG["height"])  # type: ignore[call-overload]
        )
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        logger.info("GUI thread: OpenCV window created")

    def gui_loop(self) -> None:
        """Main synchronous GUI loop - simplified to focus on coordination."""
        logger.info("Starting GUI thread loop")
        self.setup_window()

        try:
            while not self.shutdown_coordinator.is_shutdown_requested():
                # 📊 GUI FPS TRACKING: Start frame
                current_time = time.time()
                if float(self.gui_fps_tracker["last_frame_time"]) > 0:
                    frame_time = current_time - float(self.gui_fps_tracker["last_frame_time"])
                    self.gui_fps_tracker["frame_times"].append(frame_time)

                    # Maintain sliding window
                    frame_times_list = self.gui_fps_tracker["frame_times"]
                    if len(frame_times_list) > int(self.gui_fps_tracker["window_size"]):
                        frame_times_list.pop(0)

                self.gui_fps_tracker["last_frame_time"] = current_time

                # Calculate and update GUI FPS
                gui_fps = self._calculate_gui_fps()
                self.data_manager.update_gui_fps(gui_fps)

                # 🔒 THREAD-SAFE READ from shared data
                display_data = self.data_manager.get_latest_data()

                # 🚀 FRAME CACHING OPTIMIZATION - only render if new frame
                if (
                    display_data.frame is not None
                    and display_data.frame_id != self._last_rendered_frame_id
                ):
                    # Get current control state for rendering
                    control_state = self.data_manager.get_control_state()

                    # 🎨 DELEGATE TO RENDERER (pure function)
                    rendered_frame = self.renderer.create_visualization(
                        frame=display_data.frame,
                        raw_detections=display_data.raw_detections,
                        tracked_detections=display_data.tracked_detections,
                        vehicle_events=display_data.vehicle_events,
                        stats=display_data.stats,
                        control_state=control_state,
                        gui_fps=gui_fps,
                        processing_fps=display_data.processing_fps
                    )

                    # 🖥️ DISPLAY
                    cv2.imshow(self.window_name, rendered_frame)
                    self._last_rendered_frame_id = display_data.frame_id

                # 🎮 KEYBOARD INPUT HANDLING
                self._handle_keyboard_input()

        except Exception as e:
            logger.error(f"Error in GUI thread: {e}")
        finally:
            self.shutdown_coordinator.signal_gui_done()
            cv2.destroyAllWindows()
            logger.info("GUI thread finished")

    def _calculate_gui_fps(self) -> float:
        """Calculate current GUI FPS from sliding window."""
        frame_times = self.gui_fps_tracker["frame_times"]
        if len(frame_times) < 2:
            return 0.0

        avg_frame_time = sum(frame_times) / len(frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def _handle_keyboard_input(self) -> None:
        """Handle keyboard input with thread-safe control state updates."""
        key = cv2.waitKey(FPS_TRACKING_CONFIG["frame_timeout_ms"]) & 0xFF

        if key == ord("q"):
            logger.info("GUI thread: Quit requested")
            self.shutdown_coordinator.request_shutdown()
        elif key == ord("1"):
            # Toggle detection processing
            new_state = not self.data_manager.get_control_state().detection_enabled
            self.data_manager.update_control_state(detection_enabled=new_state)
            logger.info(f"🎯 Detection: {'ENABLED' if new_state else 'DISABLED'}")
        elif key == ord("2"):
            # Toggle tracking processing
            new_state = not self.data_manager.get_control_state().tracking_enabled
            self.data_manager.update_control_state(tracking_enabled=new_state)
            logger.info(f"🔄 Tracking: {'ENABLED' if new_state else 'DISABLED'}")
        elif key == ord("d"):
            # Toggle database saving
            new_state = not self.data_manager.get_control_state().database_enabled
            self.data_manager.update_control_state(database_enabled=new_state)
            logger.info(f"📊 Database saving: {'ENABLED' if new_state else 'DISABLED'}")
        elif key == ord("c"):
            # Toggle candidate image saving
            new_state = not self.data_manager.get_control_state().candidates_enabled
            self.data_manager.update_control_state(candidates_enabled=new_state)
            logger.info(f"💾 Candidate saving: {'ENABLED' if new_state else 'DISABLED'}")
        elif key == ord("t"):
            # Toggle track ID display
            new_state = not self.data_manager.get_control_state().show_track_ids
            self.data_manager.update_control_state(show_track_ids=new_state)
            logger.info(f"🏷️ Track IDs: {'SHOWN' if new_state else 'HIDDEN'}")
        elif key == ord("f"):
            # Toggle confidence display
            new_state = not self.data_manager.get_control_state().show_confidence
            self.data_manager.update_control_state(show_confidence=new_state)
            logger.info(f"📊 Confidence: {'SHOWN' if new_state else 'HIDDEN'}")
        elif key == ord("p"):
            # Toggle processing pause
            new_state = not self.data_manager.get_control_state().processing_paused
            self.data_manager.update_control_state(processing_paused=new_state)
            logger.info(f"⏯️ Processing: {'PAUSED' if new_state else 'RESUMED'}")
        elif key == ord("h"):
            display_unified_help()

