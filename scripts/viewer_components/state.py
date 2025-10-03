"""State management and thread synchronization for diagnostics viewer.

This module contains all classes responsible for thread-safe data exchange
and coordination between processing and GUI threads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray


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
    detection_enabled: bool = True
    tracking_enabled: bool = True
    show_track_ids: bool = True
    show_confidence: bool = True
    processing_paused: bool = False


class ThreadSafeDataManager:
    """Manager for thread-safe data exchange between processing and GUI threads."""

    def __init__(self) -> None:
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

    def update_control_state(self, **kwargs: Any) -> None:
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
                detection_enabled=self._control_state.detection_enabled,
                tracking_enabled=self._control_state.tracking_enabled,
                show_track_ids=self._control_state.show_track_ids,
                show_confidence=self._control_state.show_confidence,
                processing_paused=self._control_state.processing_paused,
            )


class ShutdownCoordinator:
    """Coordinates graceful shutdown between processing and GUI threads."""

    def __init__(self) -> None:
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


class DiagnosticsError(Exception):
    """Base exception for diagnostics viewer errors."""

    pass
