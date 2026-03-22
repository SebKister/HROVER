# HROVER

Python tool that overlays Garmin GPX heart rate data onto video files.

## Commands

```bash
# Install (editable, with GUI and dev deps)
pip install -e ".[gui,dev]"

# Run CLI
hrover video.mp4 activity.gpx -o output.mp4

# Run GUI
python -m hrover

# Run tests
pytest
```

## Architecture

```
src/hrover/
├── cli.py             # Argument parsing, entry point
├── gpx_parser.py      # Parse Garmin GPX HR + GPS data → HRTimeline (with .gps attached)
├── hr_data.py         # HRTimeline, GPSTimeline, HRZoneConfig — numpy-backed models
├── sync.py            # Compute video↔GPX time offset via ffprobe/file metadata
├── video_processor.py # Frame loop: read → interpolate BPM → render → mux audio
├── overlay.py         # OpenCV overlay drawing (BPM, zones, HR graph, mini-map)
└── gui.py             # PyQt6 GUI with live preview and export
```

## Processing pipeline

1. Parse GPX → `HRTimeline` with optional `GPSTimeline` attached as `.gps`
2. Detect video start time via ffprobe metadata (fallback: align GPX start to video start)
3. Compute `trace_start` / `trace_end` (video duration in GPX time) for map cropping
4. For each frame: `frame_index → UTC time → GPX time → BPM + GPS position → draw overlay`
5. Mux original audio back with ffmpeg

## Overlay components (`overlay.py`)

- **BPM + zone** — large number with zone color and Z1–Z5 label
- **HR graph** — 60 s scrolling history, line colored by zone
- **Mini-map** — GPS trace cropped to video duration; dim trace = full route, bright = traveled, orange dot = current position

`OverlayConfig` key fields: `position`, `opacity`, `graph_duration`, `show_map`, `map_size`

`auto_scale_overlay()` scales all size fields relative to video resolution (normalized to 1920 px).

## Data models (`hr_data.py`)

- `HRTimeline` — sorted HR samples, numpy interpolation, optional `.gps: GPSTimeline | None`
- `GPSTimeline` — lat/lon track with `get_position_at(t)`, `get_coords_in_range(t_start, t_end)`, `get_traveled_coords(t, t_start)`
- `HRZoneConfig` — zone thresholds and BGR colors

## HR Zones (% of max HR, default 190 bpm)

| Zone | Range  | Color  |
|------|--------|--------|
| Z1   | 50–60% | Gray   |
| Z2   | 60–70% | Blue   |
| Z3   | 70–80% | Green  |
| Z4   | 80–90% | Orange |
| Z5   | 90–100%| Red    |

## Dependencies

- `gpxpy` — GPX parsing
- `opencv-python` — video I/O and overlay rendering
- `numpy` — interpolation
- `PyQt6` — GUI (optional, install with `[gui]`)
- `ffprobe` / `ffmpeg` — metadata extraction and audio muxing (optional, system install)
