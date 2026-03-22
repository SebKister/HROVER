"""Main video processing loop: read frames, overlay HR data, write output."""

import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2

from .hr_data import HRTimeline, HRZoneConfig
from .overlay import OverlayConfig, draw_hr_overlay
from .sync import frame_to_utc


def process_video(
    video_path: str | Path,
    output_path: str | Path,
    hr_timeline: HRTimeline,
    video_start: datetime,
    offset: timedelta,
    overlay_config: OverlayConfig,
    zone_config: HRZoneConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Process a video file, overlaying HR data on each frame.

    Args:
        video_path: Path to input video.
        output_path: Path for output video (without audio).
        hr_timeline: Parsed HR data timeline.
        video_start: UTC datetime when the video starts.
        offset: Time offset (gpx_time = video_time + offset).
        overlay_config: Overlay appearance settings.
        zone_config: HR zone configuration.
        progress_callback: Called with (current_frame, total_frames).

    Returns:
        Path to the final output file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale overlay for high-res videos
    auto_scale_overlay(overlay_config, width, height)

    # Write to a temp file first, then mux audio
    temp_output = output_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer for: {temp_output}")

    # GPX time bounds for the full video (used to crop the map trace)
    trace_start = video_start + offset
    trace_end = trace_start + timedelta(seconds=total_frames / fps)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute the UTC time for this frame, then apply offset to get GPX time
            video_time = frame_to_utc(frame_idx, fps, video_start)
            gpx_time = video_time + offset

            # Get HR data
            bpm = hr_timeline.get_bpm_at(gpx_time)
            hr_window = hr_timeline.get_window(gpx_time, overlay_config.graph_duration)

            # Draw overlay
            frame = draw_hr_overlay(
                frame, bpm, hr_window, zone_config, overlay_config,
                gps_timeline=hr_timeline.gps,
                current_time=gpx_time,
                trace_start=trace_start,
                trace_end=trace_end,
            )

            writer.write(frame)
            frame_idx += 1

            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, total_frames)

    finally:
        cap.release()
        writer.release()

    # Final progress update
    if progress_callback:
        progress_callback(frame_idx, total_frames)

    # Mux audio from original video
    final_path = _mux_audio(video_path, temp_output, output_path)
    return final_path


def auto_scale_overlay(config: OverlayConfig, video_width: int, video_height: int):
    """Scale overlay dimensions relative to video resolution."""
    # Target: graph is ~25% of video width
    scale = video_width / 1920.0  # normalize to 1080p
    scale = max(0.5, min(2.0, scale))  # clamp scaling factor

    config.graph_width = int(400 * scale)
    config.graph_height = int(100 * scale)
    config.bpm_font_scale = 1.8 * scale
    config.label_font_scale = 0.5 * scale
    config.padding = int(15 * scale)
    config.map_size = int(150 * scale)


def _mux_audio(
    original_video: Path, processed_video: Path, output_path: Path
) -> Path:
    """Mux audio from original video into processed video using ffmpeg.

    If ffmpeg is not available, renames the processed video and warns the user.
    """
    if not shutil.which("ffmpeg"):
        print(
            "WARNING: ffmpeg not found. Output video will have no audio.\n"
            "Install ffmpeg and add it to PATH to preserve audio.",
            file=sys.stderr,
        )
        processed_video.rename(output_path)
        return output_path

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(processed_video),
                "-i", str(original_video),
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            processed_video.unlink(missing_ok=True)
            return output_path
        else:
            print(
                f"WARNING: ffmpeg audio mux failed: {result.stderr[:200]}\n"
                "Output video will have no audio.",
                file=sys.stderr,
            )
            processed_video.rename(output_path)
            return output_path

    except subprocess.TimeoutExpired:
        print("WARNING: ffmpeg timed out. Output video will have no audio.", file=sys.stderr)
        processed_video.rename(output_path)
        return output_path
