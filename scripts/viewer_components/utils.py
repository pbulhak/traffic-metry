"""Utilities, constants and help system for diagnostics viewer.

This module contains configuration constants, help system utilities,
and pure functions used across the diagnostics viewer components.
"""

# ============================================================================
# UNIFIED HELP SYSTEM - SINGLE SOURCE OF TRUTH FOR ALL CONTROLS
# ============================================================================

CONTROLS_HELP = {
    "q": "Quit application",
    "1": "Toggle detection processing (ON/OFF)",
    "2": "Toggle tracking processing (ON/OFF)",
    "d": "Toggle database saving (journey storage)",
    "c": "Toggle candidate image saving",
    "p": "Pause/Resume processing (GUI continues)",
    "t": "Toggle track ID display",
    "f": "Toggle confidence display",
    "h": "Show help"
}

HELP_HEADER = "TrafficMetry Interactive Diagnostics Laboratory"

HELP_DESCRIPTION = [
    "🚀 Multi-threaded architecture for optimal performance",
    "🎯 Processing & GUI threads running independently",
    "🔧 Real-time toggle controls for performance analysis",
    "⚡ Expected: ~60 FPS GUI, ~25-30 FPS processing"
]


def display_unified_help() -> None:
    """Display unified help message with all interactive controls."""
    help_text = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                    {HELP_HEADER}                     ║
        ╠══════════════════════════════════════════════════════════════╣
"""
    # Add control descriptions from unified source
    for key, description in CONTROLS_HELP.items():
        help_text += f"        ║  '{key}' - {description:<54} ║\n"

    help_text += "        ║                                                              ║\n"

    # Add feature descriptions
    for desc in HELP_DESCRIPTION:
        help_text += f"        ║  {desc:<58} ║\n"

    help_text += "        ╚══════════════════════════════════════════════════════════════╝"

    print(help_text)


def display_startup_help() -> None:
    """Display unified startup help and controls."""
    help_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    {HELP_HEADER}                     ║
    ╠══════════════════════════════════════════════════════════════╣
"""
    # Add most important controls for startup
    priority_keys = ['q', '1', '2', 'd', 'c', 'p', 'h']
    for key in priority_keys:
        if key in CONTROLS_HELP:
            help_text += f"    ║  '{key}' - {CONTROLS_HELP[key]:<54} ║\n"

    help_text += "    ║                                                              ║\n"

    # Add feature descriptions
    for desc in HELP_DESCRIPTION:
        help_text += f"    ║  {desc:<58} ║\n"

    help_text += "    ╚══════════════════════════════════════════════════════════════╝"

    print(help_text)


# ============================================================================
# COLOR CONFIGURATION
# ============================================================================

DEFAULT_COLORS = {
    "raw_detection": (100, 100, 100),  # Gray for raw detections
    "tracked_vehicle": (0, 255, 0),  # Green for tracked vehicles
    "panel_bg": (0, 0, 0),  # Black for info panels
    "stats_text": (255, 255, 255),  # White for text
    "text_shadow": (0, 0, 0),  # Black for shadows
}


# ============================================================================
# WINDOW CONFIGURATION
# ============================================================================

DEFAULT_WINDOW_CONFIG = {
    "name": "TrafficMetry Diagnostics Viewer",
    "width": 1280,
    "height": 720,
    "stats_panel_height": 120,
}


# ============================================================================
# FPS TRACKING CONFIGURATION
# ============================================================================

FPS_TRACKING_CONFIG = {
    "window_size": 30,  # Rolling average window
    "target_fps": 60,   # Target GUI FPS
    "frame_timeout_ms": 16,  # ~60 FPS timing for cv2.waitKey
}

