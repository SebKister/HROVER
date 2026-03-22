"""OpenCV overlay rendering for BPM text, scrolling HR graph, and mini-map."""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .hr_data import GPSTimeline, HRZoneConfig


@dataclass
class OverlayConfig:
    """Configuration for the HR overlay appearance and position."""
    position: str = "bottom-left"  # bottom-left, bottom-right, top-left, top-right
    graph_width: int = 400
    graph_height: int = 100
    bpm_font_scale: float = 1.8
    opacity: float = 0.7
    graph_duration: float = 60.0  # seconds of HR history to show
    bpm_range: tuple[int, int] = (40, 200)  # y-axis range for graph
    padding: int = 15
    label_font_scale: float = 0.5
    show_map: bool = True
    map_size: int = 150  # side length of the square mini-map in pixels


def draw_hr_overlay(
    frame: np.ndarray,
    bpm: float | None,
    hr_window: list[tuple[float, float]],
    zone_config: HRZoneConfig,
    config: OverlayConfig,
    gps_timeline: Optional[GPSTimeline] = None,
    current_time: Optional[datetime] = None,
    trace_start: Optional[datetime] = None,
    trace_end: Optional[datetime] = None,
) -> np.ndarray:
    """Draw the full HR overlay (BPM number + graph + optional mini-map) onto a video frame.

    Returns the modified frame.
    """
    if bpm is None:
        return frame

    h, w = frame.shape[:2]
    bpm_int = int(round(bpm))
    zone_color = zone_config.get_zone_color(bpm_int)

    show_map = (
        config.show_map
        and gps_timeline is not None
        and current_time is not None
    )

    # Calculate total overlay dimensions
    bpm_text = str(bpm_int)
    label_text = "BPM"

    # Measure BPM text size
    (bpm_w, bpm_h), bpm_baseline = cv2.getTextSize(
        bpm_text, cv2.FONT_HERSHEY_SIMPLEX, config.bpm_font_scale, 3
    )
    (label_w, label_h), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, config.label_font_scale, 1
    )

    # Total overlay area: BPM on top, graph below, optional map at bottom
    bpm_area_height = bpm_h + label_h + 10
    total_height = bpm_area_height + config.graph_height + config.padding * 3
    if show_map:
        total_height += config.map_size + config.padding
    total_width = max(config.graph_width, bpm_w + label_w + 15) + config.padding * 2
    if show_map:
        total_width = max(total_width, config.map_size + config.padding * 2)

    # Position the overlay
    x, y = _get_overlay_position(config.position, w, h, total_width, total_height, margin=20)

    # Clamp to frame bounds
    x = max(0, min(x, w - total_width))
    y = max(0, min(y, h - total_height))

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + total_width, y + total_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, config.opacity, frame, 1 - config.opacity, 0, frame)

    # Draw BPM text with shadow
    bpm_x = x + config.padding
    bpm_y = y + config.padding + bpm_h
    _draw_text_with_shadow(
        frame, bpm_text, (bpm_x, bpm_y),
        cv2.FONT_HERSHEY_SIMPLEX, config.bpm_font_scale, zone_color, 3
    )

    # Draw "BPM" label next to the number
    label_x = bpm_x + bpm_w + 8
    label_y = bpm_y
    _draw_text_with_shadow(
        frame, label_text, (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, config.label_font_scale, (200, 200, 200), 1
    )

    # Draw zone indicator
    zone = zone_config.get_zone(bpm_int)
    zone_text = f"Z{zone}" if zone > 0 else ""
    if zone_text:
        zone_x = bpm_x + bpm_w + 8
        zone_y = bpm_y - label_h - 5
        _draw_text_with_shadow(
            frame, zone_text, (zone_x, zone_y),
            cv2.FONT_HERSHEY_SIMPLEX, config.label_font_scale, zone_color, 2
        )

    # Draw HR graph
    graph_x = x + config.padding
    graph_y = y + config.padding + bpm_area_height + config.padding
    _draw_hr_graph(
        frame, hr_window, zone_config, config,
        graph_x, graph_y, config.graph_width, config.graph_height
    )

    # Draw mini-map
    if show_map:
        map_x = x + (total_width - config.map_size) // 2
        map_y = graph_y + config.graph_height + config.padding
        _draw_map(frame, gps_timeline, current_time, config, map_x, map_y,
                  trace_start=trace_start, trace_end=trace_end)

    return frame


def _get_overlay_position(
    position: str, frame_w: int, frame_h: int,
    overlay_w: int, overlay_h: int, margin: int
) -> tuple[int, int]:
    """Calculate top-left corner of overlay based on position string."""
    if position == "bottom-left":
        return margin, frame_h - overlay_h - margin
    elif position == "bottom-right":
        return frame_w - overlay_w - margin, frame_h - overlay_h - margin
    elif position == "top-left":
        return margin, margin
    elif position == "top-right":
        return frame_w - overlay_w - margin, margin
    else:
        return margin, frame_h - overlay_h - margin  # default bottom-left


def _draw_text_with_shadow(
    frame: np.ndarray, text: str, org: tuple[int, int],
    font: int, scale: float, color: tuple[int, int, int], thickness: int
):
    """Draw text with a dark shadow for readability against any background."""
    # Shadow (offset by 2px)
    shadow_org = (org[0] + 2, org[1] + 2)
    cv2.putText(frame, text, shadow_org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _draw_hr_graph(
    frame: np.ndarray,
    hr_window: list[tuple[float, float]],
    zone_config: HRZoneConfig,
    config: OverlayConfig,
    x: int, y: int, width: int, height: int,
):
    """Draw the scrolling HR line graph."""
    if len(hr_window) < 2:
        return

    bpm_min, bpm_max = config.bpm_range
    bpm_range = bpm_max - bpm_min

    # Draw subtle grid lines at zone boundaries
    for frac in (0.5, 0.6, 0.7, 0.8, 0.9):
        zone_bpm = int(zone_config.max_hr * frac)
        if bpm_min <= zone_bpm <= bpm_max:
            gy = y + height - int((zone_bpm - bpm_min) / bpm_range * height)
            cv2.line(frame, (x, gy), (x + width, gy), (60, 60, 60), 1, cv2.LINE_AA)

    # Convert data points to pixel coordinates
    points = []
    for offset_sec, bpm in hr_window:
        px = x + int(offset_sec / config.graph_duration * width)
        py = y + height - int((bpm - bpm_min) / bpm_range * height)
        py = max(y, min(y + height, py))  # clamp
        points.append((px, py, bpm))

    # Draw the line segments colored by zone
    for i in range(len(points) - 1):
        x1, y1, bpm1 = points[i]
        x2, y2, bpm2 = points[i + 1]
        avg_bpm = (bpm1 + bpm2) / 2
        color = zone_config.get_zone_color(int(avg_bpm))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Draw current position dot
    if points:
        last_x, last_y, last_bpm = points[-1]
        color = zone_config.get_zone_color(int(last_bpm))
        cv2.circle(frame, (last_x, last_y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (last_x, last_y), 4, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_map(
    frame: np.ndarray,
    gps_timeline: GPSTimeline,
    current_time: datetime,
    config: OverlayConfig,
    x: int,
    y: int,
    trace_start: Optional[datetime] = None,
    trace_end: Optional[datetime] = None,
):
    """Draw a mini-map with the GPS trace cropped to the video duration."""
    if trace_start is not None and trace_end is not None:
        all_coords = gps_timeline.get_coords_in_range(trace_start, trace_end)
    else:
        all_coords = gps_timeline.all_coords
    if len(all_coords) < 2:
        return

    current_pos = gps_timeline.get_position_at(current_time)
    if current_pos is None:
        return

    lats = [c[0] for c in all_coords]
    lons = [c[1] for c in all_coords]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # Correct longitude range for latitude distortion
    cos_lat = math.cos(math.radians((lat_min + lat_max) / 2))
    lon_range_corrected = lon_range * cos_lat
    max_range = max(lat_range, lon_range_corrected)

    if max_range == 0:
        # Stationary: just draw a dot in the center
        cx = x + config.map_size // 2
        cy = y + config.map_size // 2
        cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1, cv2.LINE_AA)
        return

    # Fit trace into the map area with padding
    pad = max(6, config.padding // 2)
    draw_size = config.map_size - 2 * pad
    scale = draw_size / max_range

    # Center the trace within the square
    x_draw = lon_range_corrected * scale
    y_draw = lat_range * scale
    x_off = pad + (draw_size - x_draw) / 2
    y_off = pad + (draw_size - y_draw) / 2

    def project(lat: float, lon: float) -> tuple[int, int]:
        px = int(x + x_off + (lon - lon_min) * cos_lat * scale)
        py = int(y + y_off + (lat_max - lat) * scale)
        return px, py

    # Full trace (dim)
    pts = [project(lat, lon) for lat, lon in all_coords]
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], (80, 80, 80), 1, cv2.LINE_AA)

    # Traveled portion (bright)
    traveled = gps_timeline.get_traveled_coords(current_time, t_start=trace_start)
    tpts = [project(lat, lon) for lat, lon in traveled]
    for i in range(len(tpts) - 1):
        cv2.line(frame, tpts[i], tpts[i + 1], (210, 210, 210), 2, cv2.LINE_AA)

    # Current position dot
    dot_r = max(3, config.map_size // 40)
    cx, cy = project(*current_pos)
    cv2.circle(frame, (cx, cy), dot_r, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), dot_r, (255, 255, 255), 1, cv2.LINE_AA)
