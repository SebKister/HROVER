"""Command-line interface for HROVER."""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

from . import __version__
from .gpx_parser import parse_gpx_to_timeline
from .hr_data import HRZoneConfig
from .overlay import OverlayConfig
from .sync import compute_offset, get_video_creation_time
from .video_processor import ENCODERS, QUALITY_LABELS, EncoderConfig, process_video


def _progress_bar(current: int, total: int):
    """Print a progress bar to stderr."""
    if total <= 0:
        return
    pct = min(100, int(current / total * 100))
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\rProcessing: [{bar}] {pct}% ({current}/{total} frames)", end="", file=sys.stderr)
    if current >= total:
        print(file=sys.stderr)  # newline at end


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="hrover",
        description="Overlay Garmin GPX heart rate data onto a video file.",
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("gpx", help="Path to GPX file with heart rate data")
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: <input>_hr.mp4)",
    )
    parser.add_argument(
        "--max-hr", type=int, default=190,
        help="Maximum heart rate for zone calculation (default: 190)",
    )
    parser.add_argument(
        "--offset-seconds", type=float, default=None,
        help="Manual time offset in seconds (positive = GPX is ahead of video). "
             "Overrides automatic sync.",
    )
    parser.add_argument(
        "--position", default="bottom-left",
        choices=["bottom-left", "bottom-right", "top-left", "top-right"],
        help="Overlay position (default: bottom-left)",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.7,
        help="Overlay background opacity 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--graph-duration", type=float, default=60.0,
        help="Seconds of HR history to show in graph (default: 60)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Skip audio muxing (faster, but output has no audio)",
    )
    parser.add_argument(
        "--encoder", default="h264",
        choices=list(ENCODERS.keys()),
        metavar="ENCODER",
        help=(
            "Video encoder for output (default: h264). "
            "Choices: " + ", ".join(f"{k} ({v['label']})" for k, v in ENCODERS.items())
        ),
    )
    parser.add_argument(
        "--quality", default="medium",
        choices=list(QUALITY_LABELS.keys()),
        metavar="QUALITY",
        help=(
            "Encoding quality preset (default: medium). "
            "Choices: " + ", ".join(f"{k} — {v}" for k, v in QUALITY_LABELS.items())
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    video_path = Path(args.video)
    gpx_path = Path(args.gpx)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not gpx_path.exists():
        print(f"Error: GPX file not found: {gpx_path}", file=sys.stderr)
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.with_stem(video_path.stem + "_hr").with_suffix(".mp4")

    # Parse GPX
    print(f"Parsing GPX: {gpx_path}")
    try:
        hr_timeline = parse_gpx_to_timeline(gpx_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  HR data: {len(hr_timeline.samples)} samples")
    print(f"  Time range: {hr_timeline.start_time} to {hr_timeline.end_time}")
    print(f"  Duration: {hr_timeline.duration}")

    # Determine sync
    if args.offset_seconds is not None:
        offset = timedelta(seconds=args.offset_seconds)
        # Use GPX start as reference; video_start = gpx_start - offset
        video_start = hr_timeline.start_time - offset
        print(f"  Manual offset: {args.offset_seconds:+.1f}s")
    else:
        video_start = get_video_creation_time(video_path)
        if video_start is None:
            print(
                "Error: Could not determine video creation time. "
                "Use --offset-seconds to specify manually.",
                file=sys.stderr,
            )
            sys.exit(1)
        offset = compute_offset(hr_timeline.start_time, video_start)
        print(f"  Video start: {video_start}")
        print(f"  Auto offset: {offset.total_seconds():+.1f}s")

    # Config
    zone_config = HRZoneConfig(max_hr=args.max_hr)
    overlay_config = OverlayConfig(
        position=args.position,
        opacity=args.opacity,
        graph_duration=args.graph_duration,
    )
    encoder_config = EncoderConfig(encoder=args.encoder, quality=args.quality)

    # Process
    print(f"\nProcessing video: {video_path}")
    print(f"Output: {output_path}")
    print(
        f"Encoder: {ENCODERS[args.encoder]['label']}, "
        f"Quality: {QUALITY_LABELS[args.quality]}"
    )

    try:
        final_path = process_video(
            video_path=video_path,
            output_path=output_path,
            hr_timeline=hr_timeline,
            video_start=video_start,
            offset=offset,
            overlay_config=overlay_config,
            zone_config=zone_config,
            progress_callback=_progress_bar,
            encoder_config=encoder_config,
        )
        print(f"\nDone! Output saved to: {final_path}")
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
