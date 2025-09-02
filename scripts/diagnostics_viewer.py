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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import ALL TrafficMetry components
from backend.camera_stream import CameraConnectionError, CameraStream
from backend.config import ModelSettings, Settings, get_config
from backend.database import DatabaseError, EventDatabase
from backend.detection_models import DetectionResult, VehicleType
from backend.detector import DetectionError, ModelLoadError, VehicleDetector
from backend.tracker import VehicleTrackingManager
from backend.vehicle_events import VehicleEntered, VehicleEvent, VehicleExited, VehicleUpdated

# Import classes from main.py (temporary until moved to backend modules)
from main import CandidateSaver

# Configure diagnostics logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("diagnostics.log"),
        logging.StreamHandler(sys.stdout)
    ]
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
    frame_times: List[float] = field(default_factory=list)
    
    # Frame storage for pause mode
    last_frame: Optional[NDArray] = None


class DiagnosticsError(Exception):
    """Base exception for diagnostics viewer errors."""
    pass


class DiagnosticsViewer:
    """Advanced TrafficMetry diagnostics with interactive OpenCV GUI."""
    
    def __init__(self, config: Settings):
        """Initialize diagnostics viewer with full TrafficMetry stack.
        
        Args:
            config: TrafficMetry configuration settings
        """
        self.config = config
        self.state = DiagnosticsState()
        self.running = False
        
        # UI constants
        self.WINDOW_NAME = "TrafficMetry Diagnostics Viewer"
        self.STATS_PANEL_HEIGHT = 120
        self.COLORS = self._define_color_scheme()
        
        # Initialize ALL components exactly like TrafficMetryProcessor
        self._initialize_components()
        
        # Database connection state
        self._db_connected = False
        
        logger.info("DiagnosticsViewer initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all TrafficMetry components."""
        try:
            # Camera stream
            self.camera = CameraStream(self.config.camera)
            
            # Vehicle detector with lazy loading
            self.detector = VehicleDetector(self.config.model)
            
            # Vehicle tracking manager with ByteTrack
            self.vehicle_tracking_manager = VehicleTrackingManager(
                track_activation_threshold=0.5,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
                update_interval_seconds=1.0
            )
            
            # LaneAnalyzer removed - dynamic direction detection now handled in VehicleTrackingManager
            
            # Candidate saver
            self.candidate_saver = CandidateSaver(
                output_dir=Path("data/unlabeled_images"),
                model_config=self.config.model
            )
            
            # Event database
            self.event_database = EventDatabase(self.config.database)
            
            logger.info("All TrafficMetry components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise DiagnosticsError(f"Component initialization failed: {e}")

    def _define_color_scheme(self) -> Dict[str, Any]:
        """Define comprehensive color scheme for visualization."""
        return {
            'lane_boundary': (0, 255, 255),      # Cyan for lane lines
            'lane_text': (255, 255, 255),        # White for text
            'panel_bg': (0, 0, 0),               # Black for panels
            'stats_text': (0, 255, 0),           # Green for stats
            'event_highlight': (0, 0, 255),      # Red for events
            'raw_detection': (100, 100, 100),    # Gray for raw detections
            'vehicle_types': {
                VehicleType.CAR: (255, 0, 0),         # Blue for cars
                VehicleType.TRUCK: (0, 255, 0),       # Green for trucks
                VehicleType.BUS: (0, 165, 255),       # Orange for buses
                VehicleType.MOTORCYCLE: (255, 0, 255), # Magenta for motorcycles
                VehicleType.BICYCLE: (255, 255, 0),   # Cyan for bicycles
                VehicleType.OTHER_VEHICLE: (128, 128, 128)  # Gray for others
            },
            'default_vehicle': (255, 255, 255),  # White fallback
            'text_shadow': (0, 0, 0)             # Black for text shadows
        }

    def _setup_opencv_window(self) -> None:
        """Setup OpenCV window with proper configuration."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)
        
        # Set window properties for better experience
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        
        logger.info(f"OpenCV window '{self.WINDOW_NAME}' created")

    async def run(self) -> None:
        """Main diagnostics loop with OpenCV integration."""
        self._setup_opencv_window()
        self.state.start_time = time.time()
        self.running = True
        
        try:
            # Connect database if enabled
            if self.state.database_enabled:
                await self.event_database.connect()
                self._db_connected = True
                logger.info("Database connected")
            
            with self.camera:
                logger.info("Diagnostics viewer started - Camera connected")
                self._display_controls_help()
                
                while self.running:
                    loop_start = time.time()
                    
                    # 1. FRAME CAPTURE
                    frame = await self._capture_frame()
                    if frame is None:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # 2. DETECTION PIPELINE (if not paused)
                    raw_detections: List[DetectionResult] = []
                    tracked_detections: List[DetectionResult] = []
                    vehicle_events: List[VehicleEvent] = []
                    
                    if not self.state.paused:
                        try:
                            # Detect vehicles
                            raw_detections = self.detector.detect_vehicles(frame)
                            self.state.total_detections += len(raw_detections)
                            
                            # Update tracking with dynamic direction detection
                            tracked_detections, vehicle_events = self.vehicle_tracking_manager.update(
                                raw_detections
                            )
                            
                            # Process events (database, logging)
                            await self._process_vehicle_events(vehicle_events)
                            
                            # Save candidates if enabled
                            if self.state.candidates_enabled:
                                await self._save_candidates(frame, tracked_detections)
                            
                            # Store frame for pause mode
                            self.state.last_frame = frame.copy()
                            
                        except DetectionError as e:
                            logger.warning(f"Detection failed: {e}")
                            raw_detections = []
                        except Exception as e:
                            logger.error(f"Processing error: {e}")
                    
                    # 3. VISUALIZATION PIPELINE
                    display_frame = await self._create_visualization(
                        frame, raw_detections, tracked_detections, vehicle_events
                    )
                    
                    # 4. SHOW FRAME AND HANDLE INPUT
                    cv2.imshow(self.WINDOW_NAME, display_frame)
                    if not await self._handle_keyboard_input():
                        break
                    
                    # 5. PERFORMANCE MONITORING
                    self._update_performance_stats(loop_start)
                    
                    # Small delay to prevent CPU overload
                    await asyncio.sleep(0.01)
                    
        except CameraConnectionError as e:
            logger.error(f"Camera error: {e}")
        except Exception as e:
            logger.error(f"Fatal error in diagnostics loop: {e}")
        finally:
            await self._cleanup()

    async def _capture_frame(self) -> Optional[NDArray]:
        """Capture frame from camera with pause mode support."""
        if not self.state.paused:
            frame = self.camera.get_frame()
            if frame is not None:
                self.state.total_frames += 1
            return frame
        else:
            # In pause mode, return last captured frame
            return self.state.last_frame

    # _assign_lanes() method removed - dynamic direction detection now handled in VehicleTrackingManager

    async def _process_vehicle_events(self, events: List[VehicleEvent]) -> None:
        """Process vehicle lifecycle events with conditional database saving."""
        for event in events:
            if isinstance(event, VehicleEntered):
                logger.info(f"🚗 Vehicle {event.journey_id} (Track {event.track_id}) ({event.vehicle_type.value}) entered")
                
            elif isinstance(event, VehicleUpdated):
                logger.debug(f"📍 Vehicle {event.journey_id} (Track {event.track_id}) updated - direction: {event.movement_direction}")
                
            elif isinstance(event, VehicleExited):
                logger.info(
                    f"🏁 Vehicle {event.journey_id} (Track {event.track_id}) journey completed: "
                    f"{event.journey.journey_duration_seconds:.1f}s, "
                    f"{event.journey.total_detections} detections, "
                    f"direction: {event.journey.movement_direction} ({event.journey.direction_confidence:.2f})"
                )
                
                # Save to database only if enabled
                if self.state.database_enabled and self._db_connected:
                    try:
                        success = await self.event_database.save_vehicle_journey(event.journey)
                        if not success:
                            logger.error(f"Failed to save journey for vehicle {event.track_id}")
                    except DatabaseError as e:
                        logger.error(f"Database error: {e}")

    async def _save_candidates(self, frame: NDArray, detections: List[DetectionResult]) -> None:
        """Save candidate images if enabled."""
        for detection in detections:
            if detection.confidence >= self.config.model.confidence_threshold:
                try:
                    saved_path = self.candidate_saver.save_candidate(frame, detection)
                    if saved_path:
                        self.state.candidates_saved += 1
                except Exception as e:
                    logger.error(f"Candidate saving error: {e}")

    async def _create_visualization(
        self,
        frame: NDArray,
        raw_detections: List[DetectionResult],
        tracked_detections: List[DetectionResult],
        events: List[VehicleEvent]
    ) -> NDArray:
        """Create multi-layer visualization with all diagnostic elements."""
        
        # Start with original frame
        display_frame = frame.copy()
        
        # Layer 1: Lane boundaries removed - no more static lane visualization
        
        # Layer 2: Raw detections (lighter overlay)
        if raw_detections:
            display_frame = self._draw_raw_detections(display_frame, raw_detections)
        
        # Layer 3: Tracked vehicles (primary overlay)
        if tracked_detections:
            display_frame = self._draw_tracked_vehicles(display_frame, tracked_detections)
        
        # Layer 4: Event highlights (entry/exit animations)
        if events:
            display_frame = self._draw_event_highlights(display_frame, events)
        
        # Layer 5: Statistics dashboard (always on top)
        display_frame = self._draw_statistics_panel(display_frame)
        
        # Layer 6: Control hints overlay
        display_frame = self._draw_controls_overlay(display_frame)
        
        return display_frame

    # _draw_lane_boundaries() method removed - no more static lane visualization

    def _draw_raw_detections(self, frame: NDArray, detections: List[DetectionResult]) -> NDArray:
        """Draw raw detections with lighter overlay."""
        for detection in detections:
            # Light gray bounding box for raw detections
            cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2),
                         self.COLORS['raw_detection'], 1)
            
            # Small centroid dot
            cv2.circle(frame, detection.centroid, 2, self.COLORS['raw_detection'], -1)
        
        return frame

    def _draw_tracked_vehicles(self, frame: NDArray, detections: List[DetectionResult]) -> NDArray:
        """Draw tracked vehicles with enhanced visualization."""
        
        for detection in detections:
            # Color coding by vehicle type
            vehicle_types_colors = self.COLORS['vehicle_types']
            color = vehicle_types_colors.get(
                detection.vehicle_type, self.COLORS['default_vehicle']
            )
            
            # Main bounding box
            cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2),
                         color, 2)
            
            # Track ID (if enabled and available)
            if self.state.show_track_ids and hasattr(detection, 'track_id') and detection.track_id is not None:
                track_label = f"ID: {detection.track_id}"
                cv2.putText(frame, track_label, (detection.x1 + 2, detection.y1 - 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text_shadow'], 2)
                cv2.putText(frame, track_label, (detection.x1, detection.y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Vehicle type and confidence
            label = detection.vehicle_type.value
            if self.state.show_confidence:
                label += f" {detection.confidence:.2f}"
            
            # Main label with shadow
            cv2.putText(frame, label, (detection.x1 + 2, detection.y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text_shadow'], 2)
            cv2.putText(frame, label, (detection.x1, detection.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Centroid dot
            cv2.circle(frame, detection.centroid, 3, color, -1)
        
        return frame

    def _draw_event_highlights(self, frame: NDArray, events: List[VehicleEvent]) -> NDArray:
        """Draw event highlights for vehicle lifecycle events."""
        for event in events:
            if isinstance(event, VehicleEntered):
                # Green circle for vehicle entry
                if hasattr(event, 'detection'):
                    center = event.detection.centroid
                    cv2.circle(frame, center, 20, (0, 255, 0), 3)
                    cv2.putText(frame, "ENTERED", (center[0] - 30, center[1] - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            elif isinstance(event, VehicleExited):
                # Red circle for vehicle exit (use best bbox center)
                if hasattr(event, 'journey') and event.journey.best_bbox:
                    x1, y1, x2, y2 = event.journey.best_bbox
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(frame, center, 20, (0, 0, 255), 3)
                    cv2.putText(frame, "EXITED", (center[0] - 25, center[1] - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def _draw_statistics_panel(self, frame: NDArray) -> NDArray:
        """Draw comprehensive statistics dashboard."""
        height, width = frame.shape[:2]
        
        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, self.STATS_PANEL_HEIGHT),
                     self.COLORS['panel_bg'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Update tracking stats
        tracking_stats = self.vehicle_tracking_manager.get_tracking_stats()
        self.state.active_vehicles = tracking_stats['active_vehicles']
        self.state.total_journeys = tracking_stats['total_journeys_completed']
        
        # Statistics text lines
        stats_lines = [
            f"FPS: {self.state.fps:.1f} | Frames: {self.state.total_frames} | Detections: {self.state.total_detections}",
            f"Active: {self.state.active_vehicles} | Journeys: {self.state.total_journeys} | Candidates: {self.state.candidates_saved}",
            f"DB: {'ON' if self.state.database_enabled else 'OFF'} | " +
            f"Candidates: {'ON' if self.state.candidates_enabled else 'OFF'} | " +
            f"Paused: {'YES' if self.state.paused else 'NO'}"
        ]
        
        # Draw stats lines
        y_offset = 25
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.COLORS['stats_text'], 2)
        
        return frame

    def _draw_controls_overlay(self, frame: NDArray) -> NDArray:
        """Draw control hints overlay."""
        height, width = frame.shape[:2]
        
        # Control hints in bottom right
        controls_text = "Press 'h' for help | 'q' to quit"
        text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        x = width - text_size[0] - 10
        y = height - 10
        
        # Background rectangle
        cv2.rectangle(frame, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5),
                     self.COLORS['panel_bg'], -1)
        
        # Text
        cv2.putText(frame, controls_text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['stats_text'], 1)
        
        return frame

    async def _handle_keyboard_input(self) -> bool:
        """Handle interactive keyboard controls."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            logger.info("Quit command received")
            return False
        
        elif key == ord('d'):
            self.state.database_enabled = not self.state.database_enabled
            status = "ENABLED" if self.state.database_enabled else "DISABLED"
            logger.info(f"Database storage {status}")
            
            # Connect/disconnect database
            if self.state.database_enabled and not self._db_connected:
                try:
                    await self.event_database.connect()
                    self._db_connected = True
                    logger.info("Database connected")
                except DatabaseError as e:
                    logger.error(f"Database connection failed: {e}")
                    self.state.database_enabled = False
        
        elif key == ord('c'):
            self.state.candidates_enabled = not self.state.candidates_enabled
            status = "ENABLED" if self.state.candidates_enabled else "DISABLED"
            logger.info(f"Candidate image saving {status}")
        
        elif key == ord('p'):
            self.state.paused = not self.state.paused
            status = "PAUSED" if self.state.paused else "RESUMED"
            logger.info(f"Processing {status}")
        
        elif key == ord('r'):
            # 'r' for reset tracking statistics
            tracking_stats = self.vehicle_tracking_manager.get_tracking_stats()
            logger.info(f"Current tracking stats: {tracking_stats}")
            # Could add functionality to reset stats here if needed
        
        elif key == ord('t'):
            self.state.show_track_ids = not self.state.show_track_ids
            logger.info(f"Track IDs {'ON' if self.state.show_track_ids else 'OFF'}")
        
        elif key == ord('f'):
            self.state.show_confidence = not self.state.show_confidence
            logger.info(f"Confidence display {'ON' if self.state.show_confidence else 'OFF'}")
        
        elif key == ord('h'):
            self._display_controls_help()
        
        return True

    def _display_controls_help(self) -> None:
        """Display keyboard control help in terminal."""
        help_text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           DIAGNOSTICS VIEWER CONTROLS (REFACTORED)          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  'q' - Quit application                                      ║
    ║  'd' - Toggle database saving (ON/OFF)                      ║
    ║  'c' - Toggle candidate image saving (ON/OFF)               ║
    ║  'p' - Pause/Resume processing                               ║
    ║  'r' - Show current tracking statistics                      ║
    ║  't' - Toggle track ID display                               ║
    ║  'f' - Toggle confidence display                             ║
    ║  'h' - Show this help again                                  ║
    ║                                                              ║
    ║  🎯 NEW: Dynamic direction detection (no lane calibration)  ║
    ║  📊 Enhanced journey analytics with movement tracking       ║
    ╚══════════════════════════════════════════════════════════════╝
        """
        print(help_text)

    def _update_performance_stats(self, loop_start_time: float) -> None:
        """Update FPS and performance statistics."""
        current_time = time.time()
        loop_duration = current_time - loop_start_time
        
        # Store recent frame times for FPS calculation
        self.state.frame_times.append(loop_duration)
        if len(self.state.frame_times) > 30:  # Keep last 30 frames
            self.state.frame_times.pop(0)
        
        # Calculate FPS every second
        if current_time - self.state.last_fps_update >= 1.0:
            if self.state.frame_times:
                avg_frame_time = sum(self.state.frame_times) / len(self.state.frame_times)
                self.state.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            self.state.last_fps_update = current_time

    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up diagnostics viewer...")
        self.running = False
        
        if self._db_connected:
            await self.event_database.close()
        
        cv2.destroyAllWindows()
        
        # Final statistics
        self._log_final_statistics()

    def _log_final_statistics(self) -> None:
        """Log comprehensive final statistics."""
        total_time = time.time() - self.state.start_time
        avg_fps = self.state.total_frames / total_time if total_time > 0 else 0
        
        logger.info("=== DIAGNOSTICS SESSION SUMMARY ===")
        logger.info(f"Session duration: {total_time:.1f} seconds")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Total frames: {self.state.total_frames}")
        logger.info(f"Total detections: {self.state.total_detections}")
        logger.info(f"Candidates saved: {self.state.candidates_saved}")
        
        tracking_stats = self.vehicle_tracking_manager.get_tracking_stats()
        logger.info(f"Total vehicles tracked: {tracking_stats['total_vehicles_tracked']}")
        logger.info(f"Complete journeys: {tracking_stats['total_journeys_completed']}")


async def main() -> None:
    """Main entry point for diagnostics viewer."""
    
    logger.info("TrafficMetry Diagnostics Viewer starting...")
    
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Configuration loaded - Camera: {config.camera.url}")
        
        # Create and run diagnostics viewer
        diagnostics = DiagnosticsViewer(config)
        await diagnostics.run()
        
    except ModelLoadError as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)
    except CameraConnectionError as e:
        logger.error(f"Camera connection failed: {e}")
        sys.exit(1)
    except DiagnosticsError as e:
        logger.error(f"Diagnostics error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Diagnostics interrupted by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        sys.exit(1)