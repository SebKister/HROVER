"""Time synchronization between GPX data and video files."""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def get_video_creation_time(video_path: str | Path) -> Optional[datetime]:
    """Get the creation time of a video file.

    Tries ffprobe first (most reliable for MP4/MOV metadata),
    then falls back to OS file creation time.
    Returns a UTC-aware datetime, or None if unable to determine.
    """
    video_path = Path(video_path)

    # Try ffprobe
    creation_time = _get_creation_time_ffprobe(video_path)
    if creation_time is not None:
        return creation_time

    # Fallback to OS file creation time (reliable on Windows)
    return _get_creation_time_os(video_path)


def _get_creation_time_ffprobe(video_path: Path) -> Optional[datetime]:
    """Extract creation_time from video metadata using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        creation_time_str = data.get("format", {}).get("tags", {}).get("creation_time")
        if not creation_time_str:
            return None

        # Parse ISO 8601 format (e.g., "2024-03-15T14:30:00.000000Z")
        dt = datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None


def _get_creation_time_os(video_path: Path) -> Optional[datetime]:
    """Get file creation time from OS metadata."""
    try:
        # On Windows, os.path.getctime returns actual creation time
        ctime = os.path.getctime(video_path)
        return datetime.fromtimestamp(ctime, tz=timezone.utc)
    except OSError:
        return None


def compute_offset(gpx_start: datetime, video_start: datetime) -> timedelta:
    """Compute the time offset between GPX and video start times.

    Returns offset such that: gpx_time = video_time + offset
    """
    return gpx_start - video_start


def frame_to_utc(frame_index: int, fps: float, video_start: datetime) -> datetime:
    """Convert a video frame index to a UTC timestamp."""
    return video_start + timedelta(seconds=frame_index / fps)
