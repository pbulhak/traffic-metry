"""TrafficMetry Diagnostics Viewer Components.

This package contains modular components for the multi-threaded diagnostics viewer,
refactored from the original monolithic design for better maintainability.
"""

from .renderer import DiagnosticsRenderer
from .state import (
    DiagnosticsError,
    SharedControlState,
    SharedFrameData,
    ShutdownCoordinator,
    ThreadSafeDataManager,
)
from .threads import GUIThread, ProcessingThread
from .utils import (
    CONTROLS_HELP,
    HELP_DESCRIPTION,
    HELP_HEADER,
    display_startup_help,
    display_unified_help,
)

__all__ = [
    # State management
    "SharedFrameData",
    "SharedControlState",
    "ThreadSafeDataManager",
    "ShutdownCoordinator",
    "DiagnosticsError",

    # Rendering
    "DiagnosticsRenderer",

    # Threading
    "ProcessingThread",
    "GUIThread",

    # Utilities
    "CONTROLS_HELP",
    "HELP_HEADER",
    "HELP_DESCRIPTION",
    "display_unified_help",
    "display_startup_help"
]

