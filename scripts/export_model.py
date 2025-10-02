"""Export YOLO model to OpenVINO INT8 format for optimized inference.

This script converts a YOLO .pt model to OpenVINO format with INT8 quantization
for improved performance on CPU inference (~20-30% FPS boost).

Usage:
    python scripts/export_model.py --source data/models/yolov8n.pt
    python scripts/export_model.py --source models/yolo11n.pt --output models/yolo11n_openvino
    python scripts/export_model.py --source models/yolo.pt --no-int8  # Disable quantization
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_openvino(
    source_model: str,
    output_dir: str | None = None,
    int8: bool = True,
) -> Path:
    """Export YOLO model to OpenVINO format.

    Args:
        source_model: Path to source .pt model file
        output_dir: Output directory name (optional, auto-generated if None)
        int8: Enable INT8 quantization for smaller size and better performance

    Returns:
        Path to exported model directory

    Raises:
        FileNotFoundError: If source model doesn't exist
        ImportError: If ultralytics is not installed
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error("Failed to import ultralytics. Install with: pip install ultralytics")
        raise ImportError("ultralytics package not found") from e

    source_path = Path(source_model)
    if not source_path.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")

    logger.info(f"Loading YOLO model from: {source_path}")
    model = YOLO(str(source_path))

    logger.info(f"Exporting to OpenVINO format (INT8 quantization: {int8})...")
    logger.info("This may take a few minutes for quantization calibration...")

    # Export to OpenVINO - ultralytics handles path automatically
    export_path = model.export(format="openvino", int8=int8)

    logger.info("Export completed successfully!")
    logger.info(f"Model saved to: {export_path}")

    # Print size comparison
    source_size_mb = source_path.stat().st_size / (1024 * 1024)
    export_path_obj = Path(export_path)
    export_size_mb = sum(f.stat().st_size for f in export_path_obj.rglob("*") if f.is_file()) / (
        1024 * 1024
    )

    logger.info("Size comparison:")
    logger.info(f"   Original .pt: {source_size_mb:.2f} MB")
    logger.info(f"   OpenVINO:     {export_size_mb:.2f} MB")
    logger.info(f"   Reduction:    {(1 - export_size_mb / source_size_mb) * 100:.1f}%")

    logger.info("\nTo use this model, update your .env file:")
    logger.info(f'   MODEL__PATH="{export_path}"')

    return Path(export_path)


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Export YOLO model to OpenVINO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export with default settings
  python scripts/export_model.py --source data/models/yolov8n.pt

  # Custom output directory
  python scripts/export_model.py --source models/yolo11n.pt --output models/yolo11n_openvino

  # Disable INT8 quantization (faster export, larger size, slightly better accuracy)
  python scripts/export_model.py --source models/yolo.pt --no-int8
        """,
    )

    parser.add_argument(
        "--source", required=True, help="Path to source .pt YOLO model file", metavar="PATH"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory name (optional, auto-generated if not specified)",
        metavar="PATH",
    )

    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Disable INT8 quantization (default: enabled for better performance)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        export_to_openvino(source_model=args.source, output_dir=args.output, int8=not args.no_int8)
        return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    except ImportError as e:
        logger.error(f"Error: {e}")
        logger.error("Install ultralytics with: pip install ultralytics")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error during export: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
