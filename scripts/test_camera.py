#!/usr/bin/env python3
"""Test script for manual verification of CameraStream functionality.

This script loads camera configuration and displays live video feed from
the RTSP stream in an OpenCV window. Press 'q' to quit.

Usage:
    python scripts/test_camera.py

Requirements:
    - Camera must be accessible at configured RTSP URL
    - Display server must be available (for cv2.imshow)
"""

import logging
import sys
import time
from pathlib import Path

import cv2

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.camera_stream import CameraConnectionError, CameraStream
from backend.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main test function for camera stream verification."""
    logger.info("Starting camera stream test")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = get_config()
        camera_settings = config.camera

        logger.info(f"Camera URL: {camera_settings.url}")
        logger.info(f"Resolution: {camera_settings.width}x{camera_settings.height}")
        logger.info(f"Target FPS: {camera_settings.fps}")

        # Initialize camera stream
        logger.info("Initializing camera stream...")

        with CameraStream(camera_settings) as camera:
            logger.info("Camera initialized successfully")
            logger.info("Press 'q' in the video window to quit")

            frame_count = 0
            start_time = time.time()

            while True:
                # Get frame from camera
                frame = camera.get_frame()

                if frame is None:
                    logger.warning("Failed to get frame, retrying...")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # Calculate and display FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Current FPS: {fps:.1f}")

                # Display frame
                cv2.imshow("TrafficMetry - Camera Test", frame)

                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit command received")
                    break

    except CameraConnectionError as e:
        logger.error(f"Camera connection error: {e}")
        if e.camera_url:
            logger.error(f"Failed URL: {e.camera_url}")
        logger.error("Please check:")
        logger.error("1. Camera is powered on and connected to network")
        logger.error("2. RTSP URL is correct in .env file")
        logger.error("3. Network connectivity to camera")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    finally:
        # Cleanup OpenCV windows
        cv2.destroyAllWindows()
        logger.info("Camera stream test completed")


if __name__ == "__main__":
    main()
