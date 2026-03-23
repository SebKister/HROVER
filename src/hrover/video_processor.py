"""Main video processing loop: read frames, overlay HR data, write output."""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2

from .hr_data import HRTimeline, HRZoneConfig
from .overlay import OverlayConfig, draw_hr_overlay
from .sync import frame_to_utc


# ---------------------------------------------------------------------------
# Encoder definitions
# ---------------------------------------------------------------------------

ENCODERS: dict[str, dict] = {
    "h264": {
        "label": "H.264 (libx264)",
        "codec": "libx264",
        "quality": {
            "low":      ["-crf", "28", "-preset", "fast"],
            "medium":   ["-crf", "23", "-preset", "medium"],
            "high":     ["-crf", "18", "-preset", "slow"],
            "lossless": ["-crf", "0",  "-preset", "medium"],
        },
    },
    "h265": {
        "label": "H.265/HEVC (libx265)",
        "codec": "libx265",
        "quality": {
            "low":      ["-crf", "32", "-preset", "fast"],
            "medium":   ["-crf", "28", "-preset", "medium"],
            "high":     ["-crf", "22", "-preset", "slow"],
            "lossless": ["-x265-params", "lossless=1", "-preset", "medium"],
        },
    },
    "vp9": {
        "label": "VP9 (libvpx-vp9)",
        "codec": "libvpx-vp9",
        "quality": {
            "low":      ["-crf", "40", "-b:v", "0"],
            "medium":   ["-crf", "31", "-b:v", "0"],
            "high":     ["-crf", "20", "-b:v", "0"],
            "lossless": ["-lossless", "1"],
        },
    },
    "av1": {
        "label": "AV1 (libaom-av1)",
        "codec": "libaom-av1",
        "quality": {
            "low":      ["-crf", "45", "-b:v", "0", "-cpu-used", "6"],
            "medium":   ["-crf", "35", "-b:v", "0", "-cpu-used", "4"],
            "high":     ["-crf", "25", "-b:v", "0", "-cpu-used", "2"],
            "lossless": ["-lossless", "1", "-cpu-used", "4"],
        },
    },
    # --- NVIDIA GPU (NVENC) encoders ---
    "h264_nvenc": {
        "label": "H.264 NVIDIA GPU (h264_nvenc)",
        "codec": "h264_nvenc",
        "quality": {
            "low":      ["-rc", "vbr", "-cq", "28", "-preset", "p4"],
            "medium":   ["-rc", "vbr", "-cq", "23", "-preset", "p4"],
            "high":     ["-rc", "vbr", "-cq", "18", "-preset", "p6"],
            "lossless": ["-preset", "lossless"],
        },
    },
    "h265_nvenc": {
        "label": "H.265/HEVC NVIDIA GPU (hevc_nvenc)",
        "codec": "hevc_nvenc",
        "quality": {
            "low":      ["-rc", "vbr", "-cq", "32", "-preset", "p4"],
            "medium":   ["-rc", "vbr", "-cq", "28", "-preset", "p4"],
            "high":     ["-rc", "vbr", "-cq", "22", "-preset", "p6"],
            "lossless": ["-preset", "lossless"],
        },
    },
    "av1_nvenc": {
        "label": "AV1 NVIDIA GPU (av1_nvenc, RTX 40+)",
        "codec": "av1_nvenc",
        "quality": {
            "low":      ["-rc", "vbr", "-cq", "45", "-preset", "p4"],
            "medium":   ["-rc", "vbr", "-cq", "35", "-preset", "p4"],
            "high":     ["-rc", "vbr", "-cq", "25", "-preset", "p6"],
            "lossless": ["-rc", "constqp", "-qp", "0"],
        },
    },
}

QUALITY_LABELS = {
    "low":      "Low (smaller file)",
    "medium":   "Medium (balanced)",
    "high":     "High (larger file)",
    "lossless": "Lossless",
}

_nvenc_available: Optional[list[str]] = None  # cached result


def detect_nvenc_encoders() -> list[str]:
    """Return a list of NVENC encoder keys available on this machine.

    Probes ffmpeg once and caches the result.  Returns an empty list when
    ffmpeg is not installed or no NVENC encoders are found.
    """
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available

    if not shutil.which("ffmpeg"):
        _nvenc_available = []
        return _nvenc_available

    nvenc_keys = [k for k in ENCODERS if k.endswith("_nvenc")]
    available = []
    for key in nvenc_keys:
        codec = ENCODERS[key]["codec"]
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc",
                 "-t", "0.01", "-c:v", codec, "-f", "null", "-"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                available.append(key)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    _nvenc_available = available
    return _nvenc_available


@dataclass
class EncoderConfig:
    """Video encoder and quality settings."""
    encoder: str = "h264"
    quality: str = "medium"

    def get_ffmpeg_args(self) -> list[str]:
        """Return the ffmpeg codec and quality arguments.

        Raises:
            ValueError: If the encoder or quality setting is not supported.
        """
        try:
            enc = ENCODERS[self.encoder]
        except KeyError as exc:
            supported_encoders = ", ".join(sorted(ENCODERS.keys()))
            raise ValueError(
                f"Unknown encoder '{self.encoder}'. Supported encoders: {supported_encoders}"
            ) from exc

        try:
            quality_args = enc["quality"][self.quality]
        except KeyError as exc:
            supported_qualities = ", ".join(sorted(enc["quality"].keys()))
            raise ValueError(
                f"Unknown quality '{self.quality}' for encoder '{self.encoder}'. "
                f"Supported qualities: {supported_qualities}"
            ) from exc

        return ["-c:v", enc["codec"]] + quality_args
def process_video(
    video_path: str | Path,
    output_path: str | Path,
    hr_timeline: HRTimeline,
    video_start: datetime,
    offset: timedelta,
    overlay_config: OverlayConfig,
    zone_config: HRZoneConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    encoder_config: Optional[EncoderConfig] = None,
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
        encoder_config: Encoder and quality settings. Defaults to H.264/medium.

    Returns:
        Path to the final output file.
    """
    if encoder_config is None:
        encoder_config = EncoderConfig()

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

    # GPX time bounds for the full video (used to crop the map trace)
    trace_start = video_start + offset
    trace_end = trace_start + timedelta(seconds=total_frames / fps)

    use_ffmpeg = shutil.which("ffmpeg") is not None

    if use_ffmpeg:
        try:
            ffmpeg_proc = _start_ffmpeg_writer(temp_output, fps, width, height, encoder_config)
        except Exception as exc:
            cap.release()
            if temp_output.exists():
                temp_output.unlink()
            raise RuntimeError(f"Failed to start ffmpeg writer: {exc}") from exc
        writer = None
    else:
        print(
            "WARNING: ffmpeg not found. Falling back to OpenCV mp4v encoder.\n"
            "Install ffmpeg for encoder/quality control.",
            file=sys.stderr,
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create video writer for: {temp_output}")
        ffmpeg_proc = None

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

            if ffmpeg_proc is not None:
                try:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                except BrokenPipeError as exc:
                    # ffmpeg exited early (e.g. invalid codec/args, disk full, etc.).
                    # Ensure the process is cleaned up and surface a helpful error.
                    try:
                        ffmpeg_proc.stdin.close()
                    except Exception:
                        # Ignore errors while closing a broken pipe.
                        pass
                    # Wait for ffmpeg to exit and capture stderr if available.
                    try:
                        ffmpeg_proc.wait()
                    except Exception:
                        pass
                    stderr_output = None
                    if getattr(ffmpeg_proc, "stderr", None) is not None:
                        try:
                            stderr_bytes = ffmpeg_proc.stderr.read()
                            if stderr_bytes:
                                stderr_output = stderr_bytes.decode(errors="replace")
                        except Exception:
                            stderr_output = None
                    msg = "ffmpeg exited unexpectedly while writing video frames."
                    if stderr_output:
                        msg += f" ffmpeg stderr:\n{stderr_output}"
                    raise RuntimeError(msg) from exc
            else:
                writer.write(frame)

            frame_idx += 1

            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, total_frames)

    finally:
        cap.release()
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            if ffmpeg_proc.returncode != 0:
                stderr_output = None
                try:
                    stderr_bytes = ffmpeg_proc.stderr.read()
                    if stderr_bytes:
                        stderr_output = stderr_bytes.decode(errors="replace")
                except Exception:
                    pass
                msg = f"ffmpeg encoding failed with exit code {ffmpeg_proc.returncode}."
                if stderr_output:
                    msg += f" ffmpeg stderr:\n{stderr_output}"
                raise RuntimeError(msg)
        if writer is not None:
            writer.release()

    # Final progress update
    if progress_callback:
        progress_callback(frame_idx, total_frames)

    # Mux audio from original video
    final_path = _mux_audio(video_path, temp_output, output_path)
    return final_path


def _start_ffmpeg_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    encoder_config: EncoderConfig,
) -> subprocess.Popen:
    """Start an ffmpeg subprocess that reads raw BGR frames from stdin and encodes them."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        *encoder_config.get_ffmpeg_args(),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


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
