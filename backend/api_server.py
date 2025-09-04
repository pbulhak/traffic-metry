"""FastAPI server for TrafficMetry real-time vehicle monitoring.

This module provides a minimal FastAPI application with WebSocket support
for real-time vehicle event streaming.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.background_processor import background_processor
from backend.event_publisher import event_publisher

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """FastAPI lifespan context manager for startup/shutdown."""
    logger.info("FastAPI server starting up...")

    try:
        # Start background processor
        await background_processor.start()
        logger.info("Background processor started")

        yield

    except Exception as e:
        logger.error(f"Failed to start background services: {e}")
        raise

    finally:
        logger.info("FastAPI server shutting down...")
        try:
            await background_processor.stop()
            logger.info("Background processor stopped")
        except Exception as e:
            logger.error(f"Error stopping background processor: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="TrafficMetry API",
        description="Real-time vehicle detection and traffic monitoring system",
        version="2.0.0",
        lifespan=lifespan,
    )

    return app


app = create_app()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time vehicle event streaming.

    Args:
        websocket: WebSocket connection
    """
    client_id = id(websocket)
    logger.info(f"WebSocket connection attempt from client {client_id}")

    # Try to connect client
    if not await event_publisher.connect_client(websocket):
        await websocket.close(code=1013, reason="Too many connections")
        logger.warning(f"WebSocket connection rejected for client {client_id}")
        return

    try:
        # Keep connection alive
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")

    finally:
        event_publisher.disconnect_client(websocket)
        logger.debug(f"WebSocket client {client_id} cleanup completed")


# Mount the entire frontend directory to the root URL (MUST be last!)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # Development server
    uvicorn.run("backend.api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
