#!/usr/bin/env python3
"""TrafficMetry Multi-Threaded Diagnostics Viewer - Main Orchestrator.

This streamlined main file coordinates all viewer components in a clean,
modular architecture following Single Responsibility Principle.

Usage:
    python scripts/diagnostics_viewer.py

Controls:
    'q' - Quit application
    '1' - Toggle detection processing (ON/OFF)
    '2' - Toggle tracking processing (ON/OFF)
    'd' - Toggle database saving (ON/OFF)
    'c' - Toggle candidate image saving (ON/OFF)
    'p' - Pause/Resume processing
    't' - Toggle track ID display
    'f' - Toggle confidence display
    'h' - Show help in terminal
"""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.config import Settings, get_config

# Import all components from modular architecture
from scripts.viewer_components import (
    DiagnosticsRenderer,
    GUIThread,
    ProcessingThread,
    ShutdownCoordinator,
    ThreadSafeDataManager,
    display_startup_help,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("diagnostics.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class MultiThreadedDiagnosticsViewer:
    """Lightweight orchestrator for multi-threaded diagnostics viewer.

    This class coordinates all components without implementing business logic,
    following the orchestration pattern for clean architecture.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize orchestrator with dependency injection.

        Args:
            config: TrafficMetry configuration settings
        """
        self.config = config

        # Core coordination components
        self.shutdown_coordinator = ShutdownCoordinator()
        self.shared_data_manager = ThreadSafeDataManager()

        # Initialize renderer (pure rendering logic)
        self.renderer = DiagnosticsRenderer()

        # Initialize thread components with dependency injection
        self.processing_thread = ProcessingThread(
            config=self.config,
            data_manager=self.shared_data_manager,
            shutdown_coordinator=self.shutdown_coordinator,
        )

        self.gui_thread = GUIThread(
            data_manager=self.shared_data_manager,
            shutdown_coordinator=self.shutdown_coordinator,
            renderer=self.renderer,  # Dependency injection
        )

        logger.info("MultiThreadedDiagnosticsViewer initialized successfully")

    def run(self) -> None:
        """Main orchestration - start threads and manage lifecycle."""
        logger.info("TrafficMetry Multi-Threaded Diagnostics Viewer starting...")

        # Display startup information
        logger.info(f"Configuration loaded - Camera: {self.config.camera.url}")
        display_startup_help()

        # 🚀 THREAD ORCHESTRATION
        processing_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Processing")

        try:
            # Submit processing thread to executor (async in background)
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
            # 🛑 GRACEFUL SHUTDOWN COORDINATION
            logger.info("Initiating graceful shutdown...")
            self.shutdown_coordinator.request_shutdown()

            # Wait for processing thread completion
            try:
                processing_future.result(timeout=10.0)
                logger.info("Processing thread shutdown complete")
            except TimeoutError:
                logger.warning("Processing thread did not stop gracefully")
            except Exception as e:
                logger.error(f"Error during processing thread shutdown: {e}")

            # Shutdown thread pool
            processing_executor.shutdown(wait=True)

            # Verify complete shutdown
            if self.shutdown_coordinator.wait_for_complete_shutdown():
                logger.info("All threads shutdown successfully")
            else:
                logger.warning("Some threads did not shutdown gracefully")

            # 📊 DISPLAY FINAL SUMMARY
            self._display_final_summary()
            logger.info("=== MULTI-THREADED DIAGNOSTICS SESSION COMPLETE ===")

    def _display_final_summary(self) -> None:
        """Display final session statistics summary."""
        try:
            # Get final shared data and stats
            final_data = self.shared_data_manager.get_latest_data()
            control_state = self.shared_data_manager.get_control_state()

            # Get session statistics from processing thread
            session_duration = getattr(self.processing_thread, "session_duration", 0.0)
            frame_count = getattr(self.processing_thread, "frame_count", 0)
            detection_count = getattr(self.processing_thread, "detection_count", 0)
            # event_count = getattr(self.processing_thread, "event_count", 0)  # Not used

            # Calculate average FPS
            avg_fps = frame_count / session_duration if session_duration > 0 else 0.0

            # Get tracking statistics if available
            tracking_stats = final_data.stats.get("tracking_stats", {})
            total_vehicles = tracking_stats.get("total_vehicles_tracked", 0)
            complete_journeys = tracking_stats.get("total_journeys_completed", 0)

            # Get candidate statistics from event-driven candidate saver
            candidate_stats = self.processing_thread.event_candidate_saver.get_statistics()
            candidates_saved = candidate_stats.get("saved_candidates", 0)

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
