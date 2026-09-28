"""The ``pybeamprofiler`` command.

A thin layer over :class:`~pybeamprofiler.beamprofiler.BeamProfiler`: parse
the arguments, build the profiler, hand it to :meth:`BeamProfiler.plot`, and
release the camera however that ends. Also what ``python -m pybeamprofiler``
runs.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

from .beamprofiler import BeamProfiler

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI entry point for pyBeamprofiler.

    Returns:
        The process exit status: 0, or 1 if the profiler could not start.
    """
    parser = argparse.ArgumentParser(
        # Without this, ``python -m pybeamprofiler`` calls itself __main__.py.
        prog="pybeamprofiler",
        description="pyBeamprofiler - Laser beam profiler with Gaussian fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Simulated camera with continuous streaming
        pybeamprofiler

        # FLIR camera, single shot
        pybeamprofiler --camera flir --num-img 1

        # Static image file (pixel size is required — it isn't in the file)
        pybeamprofiler --file beam.png --pixel-size 5.86

        # Basler camera with 2D fitting and FWHM definition
        pybeamprofiler --camera basler --fit 2d --definition fwhm
        """,
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="simulated",
        choices=["simulated", "flir", "basler"],
        help="Camera type (default: simulated)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to static image file (overrides --camera)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help=(
            "Sensor pixel pitch in micrometers. Required with --file, since an "
            "image on disk carries no scale. Optional with a camera, where it "
            "overrides the pitch the camera reports."
        ),
    )
    parser.add_argument(
        "--fit",
        type=str,
        default="1d",
        choices=["1d", "2d", "linecut"],
        help="Fitting method: 1d (fastest), 2d (with rotation), linecut (default: 1d)",
    )
    parser.add_argument(
        "--definition",
        type=str,
        default="gaussian",
        choices=["gaussian", "fwhm", "d4s"],
        help="Width definition: gaussian (1/e²), fwhm, d4s (default: gaussian)",
    )
    parser.add_argument(
        "--exposure-time",
        type=float,
        default=None,
        help="Camera exposure time in seconds (set during initialization)",
    )

    parser.add_argument(
        "--num-img",
        type=int,
        default=None,
        help="1 for a single shot; omit to stream continuously",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Draw only the heatmap, without the profile curves",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.file and args.pixel_size is None:
        parser.error("--pixel-size is required with --file (e.g. --pixel-size 5.86)")
    if args.pixel_size is not None and not (math.isfinite(args.pixel_size) and args.pixel_size > 0):
        parser.error("--pixel-size must be a positive number")
    if args.num_img is not None and args.num_img != 1:
        parser.error("--num-img only supports 1 (a single shot); omit it to stream")

    logger.info("Initializing pyBeamprofiler...")
    logger.info(f"   Camera: {args.file if args.file else args.camera}")
    logger.info(f"   Fitting: {args.fit} ({args.definition})")

    bp: BeamProfiler | None = None
    try:
        bp = BeamProfiler(
            camera=None if args.file else args.camera,
            file=args.file,
            fit=args.fit,
            definition=args.definition,
            exposure_time=args.exposure_time,
            pixel_size=args.pixel_size,
        )

        logger.info(f"   Sensor: {bp.width_pixels}×{bp.height_pixels} pixels")
        logger.info(f"   Pixel size: {bp.pixel_size:.2f} μm")
        logger.info("Single shot acquisition..." if args.num_img == 1 else "Streaming...")

        bp.plot(num_img=args.num_img, heatmap_only=args.heatmap_only)
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
    except Exception as e:
        # A camera that isn't plugged in is the common case here; it deserves
        # a one-line message rather than a traceback, unless -v asks for one.
        # -v means INFO, so that is the level the traceback has to be at.
        logger.info("Fatal error", exc_info=True)
        print(f"pybeamprofiler: error: {e}", file=sys.stderr)
        return 1
    finally:
        if bp is not None:
            bp._release_camera()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
