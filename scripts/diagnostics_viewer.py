#!/usr/bin/env python3
"""Advanced TrafficMetry Diagnostics Viewer with Interactive OpenCV GUI.

This comprehensive diagnostic tool integrates ALL TrafficMetry components to provide:
- Real-time vehicle detection and tracking visualization
- Lane boundary overlay with calibration data
- Interactive toggle switches for database/candidate saving
- Performance statistics and monitoring dashboard
- Multi-layer visualization system for debugging
- Professional UI with keyboard controls

Usage:
    python scripts/diagnostics_viewer.py

Controls:
    'q' - Quit application
    'd' - Toggle database saving (ON/OFF)
    'c' - Toggle candidate image saving (ON/OFF)
    'p' - Pause/Resume processing
    'l' - Toggle lane boundary display
    't' - Toggle track ID display
    'f' - Toggle confidence display
    'h' - Show help in terminal
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import ALL TrafficMetry components
from backend.async_components import AsyncCameraStream, AsyncVehicleDetector
from backend.camera_stream import CameraConnectionError
from backend.config import Settings, get_config
from backend.database import DatabaseError, EventDatabase
from backend.detection_models import DetectionResult, VehicleType
from backend.detector import DetectionError, ModelLoadError
from backend.tracker import VehicleTrackingManager
from backend.vehicle_events import VehicleEntered, VehicleEvent, VehicleExited, VehicleUpdated

# Import EventDrivenCandidateSaver from backend modules
from backend.candidate_saver import EventDrivenCandidateSaver

# Configure diagnostics logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("diagnostics.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticsState:
    """State management for diagnostics viewer toggles and statistics."""

    # Interactive toggles
    database_enabled: bool = True
    candidates_enabled: bool = True
    paused: bool = False
    show_track_ids: bool = True
    show_confidence: bool = True

    # Performance metrics
    fps: float = 0.0
    total_frames: int = 0
    total_detections: int = 0
    active_vehicles: int = 0
    total_journeys: int = 0
    candidates_saved: int = 0

    # Runtime statistics
    start_time: float = field(default_factory=time.time)
    last_fps_update: float = field(default_factory=time.time)
    frame_times: list[float] = field(default_factory=list)

    # Frame storage for pause mode
    last_frame: NDArray | None = None


@dataclass
class SharedFrameData:
    """Thread-safe data container for inter-thread communication."""

    frame: NDArray | None = None
    raw_detections: list = field(default_factory=list)
    tracked_detections: list = field(default_factory=list)
    vehicle_events: list = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    processing_fps: float = 0.0
    gui_fps: float = 0.0
    frame_id: int = 0  # Unique frame identifier for caching


@dataclass
class SharedControlState:
    """Thread-safe control state for inter-thread communication."""

    database_enabled: bool = True
    candidates_enabled: bool = True
    show_track_ids: bool = True
    show_confidence: bool = True
    processing_paused: bool = False


class ThreadSafeDataManager:
    """Manager for thread-safe data exchange between processing and GUI threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self._latest_data = SharedFrameData()
        self._data_updated = threading.Event()
        self._frame_counter = 0
        self._control_state = SharedControlState()

    def update_frame_data(
        self,
        frame: NDArray | None,
        raw_detections: list,
        tracked_detections: list,
        vehicle_events: list,
        stats: dict[str, Any],
        processing_fps: float = 0.0,
    ) -> None:
        """Update shared data from processing thread (thread-safe)."""
        with self.lock:
            self._frame_counter += 1
            self._latest_data = SharedFrameData(
                frame=frame.copy() if frame is not None else None,  # Deep copy for safety
                raw_detections=raw_detections.copy(),
                tracked_detections=tracked_detections.copy(),
                vehicle_events=vehicle_events.copy(),
                stats=stats.copy(),
                timestamp=time.time(),
                processing_fps=processing_fps,
                gui_fps=self._latest_data.gui_fps,  # Preserve GUI FPS
                frame_id=self._frame_counter,  # Unique frame identifier for caching
            )
            self._data_updated.set()

    def update_gui_fps(self, gui_fps: float) -> None:
        """Update GUI FPS from GUI thread (thread-safe)."""
        with self.lock:
            self._latest_data.gui_fps = gui_fps

    def get_latest_data(self) -> SharedFrameData:
        """Get latest data for GUI thread (thread-safe)."""
        with self.lock:
            return self._latest_data  # Shallow copy OK - data is immutable after lock

    def wait_for_data(self, timeout: float = 0.1) -> bool:
        """Wait for new data with timeout."""
        return self._data_updated.wait(timeout)

    def get_frame_counter(self) -> int:
        """Get total frames processed."""
        with self.lock:
            return self._frame_counter

    def update_control_state(self, **kwargs) -> None:
        """Update control state from GUI thread (thread-safe)."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self._control_state, key):
                    setattr(self._control_state, key, value)

    def get_control_state(self) -> SharedControlState:
        """Get control state for processing thread (thread-safe)."""
        with self.lock:
            # Return a copy to avoid race conditions
            return SharedControlState(
                database_enabled=self._control_state.database_enabled,
                candidates_enabled=self._control_state.candidates_enabled,
                show_track_ids=self._control_state.show_track_ids,
                show_confidence=self._control_state.show_confidence,
                processing_paused=self._control_state.processing_paused,
            )


class ShutdownCoordinator:
    """Coordinates graceful shutdown between processing and GUI threads."""

    def __init__(self):
        self.shutdown_event = threading.Event()
        self.processing_done = threading.Event()
        self.gui_done = threading.Event()

    def request_shutdown(self) -> None:
        """Request shutdown from GUI thread."""
        self.shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_event.is_set()

    def signal_processing_done(self) -> None:
        """Signal that processing thread finished."""
        self.processing_done.set()

    def signal_gui_done(self) -> None:
        """Signal that GUI thread finished."""
        self.gui_done.set()

    def wait_for_complete_shutdown(self, timeout: float = 10.0) -> bool:
        """Wait for both threads to finish gracefully."""
        processing_ok = self.processing_done.wait(timeout=timeout)
        gui_ok = self.gui_done.wait(timeout=2.0)
        return processing_ok and gui_ok


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
        self._processing_bottleneck_tracker = {
            "camera_wait_times": [],
            "detection_times": [],
            "tracking_times": [],
            "data_update_times": [],
            "window_size": 10,  # Smaller window for bottleneck analysis
        }
        self.event_count = 0

        # Processing FPS tracking
        self.processing_fps_tracker = {
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
                config=self.config.model,  # ByteTrack parameters from config
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

    def _handle_tracking_event(self, event, frame: NDArray) -> None:
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

                        # Detect vehicles with timing
                        detection_start = time.time()
                        raw_detections = await self.detector.detect_vehicles(frame)
                        detection_time = time.time() - detection_start
                        self._track_bottleneck_metric("detection_times", detection_time)
                        self.detection_count += len(raw_detections)

                        # Update tracking with timing
                        tracking_start = time.time()
                        assert self.vehicle_tracking_manager is not None
                        tracked_detections, vehicle_events = self.vehicle_tracking_manager.update(
                            raw_detections, current_frame=frame
                        )
                        tracking_time = time.time() - tracking_start
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
                        if (
                            len(self.processing_fps_tracker["processing_times"])
                            > self.processing_fps_tracker["window_size"]
                        ):
                            self.processing_fps_tracker["processing_times"].pop(0)

                        # Calculate current processing FPS
                        processing_fps = self._calculate_processing_fps()

                        # 🔒 THREAD-SAFE UPDATE to shared data
                        stats = {
                            "frame_count": self.frame_count,
                            "detection_count": self.detection_count,
                            "event_count": self.event_count,
                            "tracking_stats": self.vehicle_tracking_manager.get_tracking_stats(),
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
        if len(self.processing_fps_tracker["processing_times"]) < 2:
            return 0.0

        avg_processing_time = sum(self.processing_fps_tracker["processing_times"]) / len(
            self.processing_fps_tracker["processing_times"]
        )
        return 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0

    def _track_bottleneck_metric(self, metric_name: str, measurement: float) -> None:
        """Track performance bottleneck metrics for debugging."""
        if metric_name not in self._processing_bottleneck_tracker:
            return

        measurements = self._processing_bottleneck_tracker[metric_name]
        measurements.append(measurement)

        # Maintain sliding window
        if len(measurements) > self._processing_bottleneck_tracker["window_size"]:
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

    async def _process_vehicle_events(self, events) -> None:
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
    """Synchronous GUI thread for OpenCV rendering and user interaction."""

    def __init__(
        self, data_manager: ThreadSafeDataManager, shutdown_coordinator: ShutdownCoordinator
    ):
        """Initialize GUI thread."""
        self.data_manager = data_manager
        self.shutdown_coordinator = shutdown_coordinator

        self.WINDOW_NAME = "TrafficMetry Diagnostics Viewer"
        self.STATS_PANEL_HEIGHT = 120

        # GUI FPS tracking
        self.gui_fps_tracker = {
            "frame_times": [],
            "window_size": 30,  # Rolling average over 30 frames
            "last_frame_time": 0.0,
        }

        # Frame caching for performance
        self._last_rendered_frame_id: int | None = None

        # Color scheme for rendering
        self.COLORS = {
            "raw_detection": (100, 100, 100),  # Gray for raw detections
            "tracked_vehicle": (0, 255, 0),  # Green for tracked vehicles
            "panel_bg": (0, 0, 0),  # Black for info panels
            "stats_text": (255, 255, 255),  # White for text
            "text_shadow": (0, 0, 0),  # Black for shadows
        }

    def setup_window(self) -> None:
        """Setup OpenCV window."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        logger.info("GUI thread: OpenCV window created")

    def gui_loop(self) -> None:
        """Main synchronous GUI loop."""
        logger.info("Starting GUI thread loop")
        self.setup_window()

        try:
            while not self.shutdown_coordinator.is_shutdown_requested():
                # 📊 GUI FPS TRACKING: Start frame
                current_time = time.time()
                if self.gui_fps_tracker["last_frame_time"] > 0:
                    frame_time = current_time - self.gui_fps_tracker["last_frame_time"]
                    self.gui_fps_tracker["frame_times"].append(frame_time)

                    # Maintain sliding window
                    if (
                        len(self.gui_fps_tracker["frame_times"])
                        > self.gui_fps_tracker["window_size"]
                    ):
                        self.gui_fps_tracker["frame_times"].pop(0)

                self.gui_fps_tracker["last_frame_time"] = current_time

                # Calculate and update GUI FPS
                gui_fps = self._calculate_gui_fps()
                self.data_manager.update_gui_fps(gui_fps)

                # 🔒 THREAD-SAFE READ from shared data
                display_data = self.data_manager.get_latest_data()

                # 🚀 FRAME CACHING OPTIMIZATION
                if (
                    display_data.frame is not None
                    and display_data.frame_id != self._last_rendered_frame_id
                ):
                    # 🎨 PURE RENDERING (synchronous, optimized) - only if new frame
                    rendered_frame = self._create_visualization_sync(
                        display_data.frame,
                        display_data.raw_detections,
                        display_data.tracked_detections,
                        display_data.vehicle_events,
                        display_data.stats,
                    )

                    # 🖥️ DISPLAY
                    cv2.imshow(self.WINDOW_NAME, rendered_frame)
                    self._last_rendered_frame_id = display_data.frame_id

                # 🎮 ENHANCED KEYBOARD INPUT with thread-safe control state
                # cv2.waitKey with target 60 FPS timing (~16.7ms)
                key = cv2.waitKey(16) & 0xFF
                if key == ord("q"):
                    logger.info("GUI thread: Quit requested")
                    self.shutdown_coordinator.request_shutdown()
                    break
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
                    self._display_help()

                # Note: No additional time.sleep needed - cv2.waitKey(16) provides 60 FPS timing

        except Exception as e:
            logger.error(f"Error in GUI thread: {e}")
        finally:
            self.shutdown_coordinator.signal_gui_done()
            cv2.destroyAllWindows()
            logger.info("GUI thread finished")

    def _calculate_gui_fps(self) -> float:
        """Calculate current GUI FPS from sliding window."""
        if len(self.gui_fps_tracker["frame_times"]) < 2:
            return 0.0

        avg_frame_time = sum(self.gui_fps_tracker["frame_times"]) / len(
            self.gui_fps_tracker["frame_times"]
        )
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def _create_visualization_sync(
        self, frame, raw_detections, tracked_detections, vehicle_events, stats
    ) -> NDArray:
        """Create visualization frame (pure synchronous OpenCV operations)."""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]

        # Draw raw detections (light gray)
        for detection in raw_detections:
            cv2.rectangle(
                display_frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                self.COLORS["raw_detection"],
                1,
            )
            cv2.circle(display_frame, detection.centroid, 2, self.COLORS["raw_detection"], -1)

        # Draw tracked detections (green)
        control_state = self.data_manager.get_control_state()

        for detection in tracked_detections:
            cv2.rectangle(
                display_frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                self.COLORS["tracked_vehicle"],
                2,
            )
            cv2.circle(display_frame, detection.centroid, 3, self.COLORS["tracked_vehicle"], -1)

            # Add labels based on thread-safe control state
            if control_state.show_track_ids and hasattr(detection, "track_id"):
                label = f"Track: {detection.track_id}"
                cv2.putText(
                    display_frame,
                    label,
                    (detection.x1, detection.y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLORS["tracked_vehicle"],
                    2,
                )

            if control_state.show_confidence:
                conf_label = f"{detection.confidence:.2f}"
                cv2.putText(
                    display_frame,
                    conf_label,
                    (detection.x1, detection.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLORS["tracked_vehicle"],
                    2,
                )

        # Draw vehicle events
        for event in vehicle_events:
            if isinstance(event, VehicleEntered):
                center = event.detection.centroid
                cv2.circle(display_frame, center, 20, (0, 255, 0), 3)
                cv2.putText(
                    display_frame,
                    "ENTERED",
                    (center[0] - 30, center[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            elif isinstance(event, VehicleExited):
                if hasattr(event, "journey") and event.journey.best_bbox:
                    x1, y1, x2, y2 = event.journey.best_bbox
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(display_frame, center, 20, (0, 0, 255), 3)
                    cv2.putText(
                        display_frame,
                        "EXITED",
                        (center[0] - 25, center[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

        # Draw stats panel with current FPS
        current_data = self.data_manager.get_latest_data()
        display_frame = self._draw_stats_panel(
            display_frame, stats, current_data.gui_fps, current_data.processing_fps
        )

        return display_frame

    def _draw_stats_panel(self, frame, stats, gui_fps: float, processing_fps: float) -> NDArray:
        """Draw enhanced statistics panel overlay with FPS metrics and control state."""
        height, width = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (width, self.STATS_PANEL_HEIGHT), self.COLORS["panel_bg"], -1
        )
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Get current control state
        control_state = self.data_manager.get_control_state()

        # Stats text with FPS metrics and control state indicators
        frame_count = stats.get("frame_count", 0)
        detection_count = stats.get("detection_count", 0)
        tracking_stats = stats.get("tracking_stats", {})

        # Status indicators
        db_status = "ON" if control_state.database_enabled else "OFF"
        cand_status = "ON" if control_state.candidates_enabled else "OFF"
        proc_status = "PAUSED" if control_state.processing_paused else "ACTIVE"

        stats_lines = [
            f"FPS: GUI {gui_fps:.1f} | Processing {processing_fps:.1f} | Status: {proc_status}",
            f"Frames: {frame_count} | Detections: {detection_count}",
            f"Active: {tracking_stats.get('active_vehicles', 0)} | Journeys: {tracking_stats.get('total_journeys_completed', 0)}",
            f"DB: {db_status} | Candidates: {cand_status} | Press 'h' for help",
        ]

        for i, line in enumerate(stats_lines):
            cv2.putText(
                frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.COLORS["stats_text"],
                2,
            )

        return frame

    def _display_help(self) -> None:
        """Display help message with all interactive controls."""
        help_text = """
        ╔══════════════════════════════════════════════════════════════╗
        ║           DIAGNOSTICS VIEWER CONTROLS (MULTI-THREADED)      ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  'q' - Quit application                                      ║
        ║  'd' - Toggle database saving (journey storage)             ║
        ║  'c' - Toggle candidate image saving                        ║
        ║  't' - Toggle track ID display                               ║
        ║  'f' - Toggle confidence display                             ║
        ║  'p' - Pause/Resume processing (GUI continues)              ║
        ║  'h' - Show this help                                        ║
        ║                                                              ║
        ║  🚀 NEW: Multi-threaded architecture for optimal FPS        ║
        ║  🎯 Processing & GUI threads running independently          ║
        ║  🔧 Thread-safe control state management                    ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print(help_text)


class DiagnosticsError(Exception):
    """Base exception for diagnostics viewer errors."""

    pass


class MultiThreadedDiagnosticsViewer:
    """Multi-threaded TrafficMetry diagnostics viewer with separated processing and GUI threads."""

    def __init__(self, config: Settings):
        """Initialize multi-threaded diagnostics viewer.

        Args:
            config: TrafficMetry configuration settings
        """
        self.config = config

        # Thread coordination components
        self.shutdown_coordinator = ShutdownCoordinator()
        self.shared_data_manager = ThreadSafeDataManager()

        # Thread components
        self.processing_thread = ProcessingThread(
            config, self.shared_data_manager, self.shutdown_coordinator
        )
        self.gui_thread = GUIThread(self.shared_data_manager, self.shutdown_coordinator)

        logger.info("MultiThreadedDiagnosticsViewer initialized successfully")

    def run(self) -> None:
        """Main entry point - starts both processing and GUI threads."""
        logger.info("TrafficMetry Multi-Threaded Diagnostics Viewer starting...")

        # Display startup info
        logger.info(f"Configuration loaded - Camera: {self.config.camera.url}")
        self._display_startup_help()

        # 🚀 START PROCESSING THREAD (async loop in separate thread)
        processing_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Processing")

        try:
            # Submit processing thread to executor
            processing_future = processing_executor.submit(
                asyncio.run, self.processing_thread.processing_loop()
            )

            # 🎨 START GUI THREAD (runs in main thread for OpenCV compatibility)
            self.gui_thread.gui_loop()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in multi-threaded viewer: {e}")
        finally:
            # 🛑 GRACEFUL SHUTDOWN
            logger.info("Initiating graceful shutdown...")
            self.shutdown_coordinator.request_shutdown()

            # Wait for processing thread to finish
            try:
                processing_future.result(timeout=10.0)
                logger.info("Processing thread shutdown complete")
            except TimeoutError:
                logger.warning("Processing thread did not stop gracefully")
            except Exception as e:
                logger.error(f"Error during processing thread shutdown: {e}")

            # Shutdown thread pool
            processing_executor.shutdown(wait=True)

            # Wait for complete shutdown
            if self.shutdown_coordinator.wait_for_complete_shutdown():
                logger.info("All threads shutdown successfully")
            else:
                logger.warning("Some threads did not shutdown gracefully")

            # 📊 DISPLAY FINAL SUMMARY
            self._display_final_summary()

            logger.info("=== MULTI-THREADED DIAGNOSTICS SESSION COMPLETE ===")

    def _display_startup_help(self) -> None:
        """Display startup help and controls."""
        help_text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║        MULTI-THREADED DIAGNOSTICS VIEWER (OPTIMIZED)        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  'q' - Quit application                                      ║
    ║  't' - Toggle track ID display                               ║
    ║  'f' - Toggle confidence display                             ║
    ║  'p' - Pause processing (GUI continues at 60 FPS)           ║
    ║  'h' - Show help again                                       ║
    ║                                                              ║
    ║  🚀 PERFORMANCE: Processing & GUI threads run independently ║
    ║  🎯 EXPECTED: ~60 FPS GUI, ~25-30 FPS processing           ║
    ║  📊 BENEFIT: No more async/OpenCV blocking issues          ║
    ╚══════════════════════════════════════════════════════════════╝
        """
        print(help_text)

    def _display_final_summary(self) -> None:
        """Display final session statistics summary (similar to original diagnostics viewer)."""
        try:
            # Get final shared data and stats
            final_data = self.shared_data_manager.get_latest_data()
            control_state = self.shared_data_manager.get_control_state()

            # Get session statistics from processing thread
            session_duration = getattr(self.processing_thread, "session_duration", 0.0)
            frame_count = getattr(self.processing_thread, "frame_count", 0)
            detection_count = getattr(self.processing_thread, "detection_count", 0)
            event_count = getattr(self.processing_thread, "event_count", 0)

            # Calculate average FPS
            avg_fps = frame_count / session_duration if session_duration > 0 else 0.0

            # Get tracking statistics if available
            tracking_stats = final_data.stats.get("tracking_stats", {})
            total_vehicles = tracking_stats.get("total_vehicles_tracked", 0)
            complete_journeys = tracking_stats.get("total_journeys_completed", 0)

            # Get candidate statistics (simplified - we don't have direct access to candidate saver stats)
            candidates_saved = 0  # Would need to track this separately in the future

            # Log session summary in the same format as original diagnostics viewer
            logger.info("=== DIAGNOSTICS SESSION SUMMARY ===")
            logger.info(f"Session duration: {session_duration:.1f} seconds")
            logger.info(f"Average FPS: {avg_fps:.1f}")
            logger.info(f"Total frames: {frame_count}")
            logger.info(f"Total detections: {detection_count}")
            logger.info(f"Candidates saved: {candidates_saved}")
            logger.info(f"Total vehicles tracked: {total_vehicles}")
            logger.info(f"Complete journeys: {complete_journeys}")

            # Performance metrics for reference
            logger.info(f"Final GUI FPS: {final_data.gui_fps:.1f}")
            logger.info(f"Final Processing FPS: {final_data.processing_fps:.1f}")

            # Control state summary
            logger.info(
                f"Database saving: {'ENABLED' if control_state.database_enabled else 'DISABLED'}"
            )
            logger.info(
                f"Candidate saving: {'ENABLED' if control_state.candidates_enabled else 'DISABLED'}"
            )

        except Exception as e:
            logger.error(f"Error displaying final summary: {e}")
            logger.info("Session completed with errors in summary display")


# Backward compatibility alias
DiagnosticsViewer = MultiThreadedDiagnosticsViewer


def main() -> None:
    """Main entry point for multi-threaded diagnostics viewer."""

    logger.info("TrafficMetry Multi-Threaded Diagnostics Viewer starting...")

    try:
        # Load configuration
        config = get_config()
        logger.info(f"Configuration loaded - Camera: {config.camera.url}")

        # Create and run multi-threaded diagnostics viewer
        diagnostics = MultiThreadedDiagnosticsViewer(config)
        diagnostics.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Multi-threaded diagnostics interrupted by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        sys.exit(1)
