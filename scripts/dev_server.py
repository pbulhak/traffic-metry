"""Development server with hot-reload functionality.

This script provides a development server for TrafficMetry with automatic
reload capabilities. It monitors Python files, environment files, and
configuration files for changes and restarts the server accordingly.
"""

from pathlib import Path

import uvicorn


def main() -> None:
    """Start the development server with hot-reload enabled.

    The server monitors the following file types for changes:
    - Python files (*.py) in backend/ directory
    - Environment files (.env) in project root
    - Configuration files (*.ini) in project root

    When any monitored file changes, the server automatically restarts.
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent

    # Development server configuration
    uvicorn.run(
        # Application module (will be created in future iterations)
        "backend.main:app",
        # Server configuration
        host="0.0.0.0",
        port=8000,
        # Hot-reload configuration
        reload=True,
        reload_dirs=[str(project_root / "backend")],
        reload_includes=["*.py", "*.env", "*.ini"],
        # Development settings
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
