"""Heart rate data model, interpolation, and zone logic."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


@dataclass
class HRSample:
    """A single heart rate measurement at a point in time."""
    timestamp: datetime
    bpm: int


@dataclass
class GPSPoint:
    """A single GPS position at a point in time."""
    timestamp: datetime
    lat: float
    lon: float


class GPSTimeline:
    """GPS track timeline with position interpolation."""

    def __init__(self, points: list[GPSPoint]):
        if not points:
            raise ValueError("GPSTimeline requires at least one point")
        self.points = sorted(points, key=lambda p: p.timestamp)
        self._epochs = np.array([p.timestamp.timestamp() for p in self.points])
        self._lats = np.array([p.lat for p in self.points])
        self._lons = np.array([p.lon for p in self.points])

    @property
    def all_coords(self) -> list[tuple[float, float]]:
        """All (lat, lon) pairs in chronological order."""
        return list(zip(self._lats.tolist(), self._lons.tolist()))

    def get_position_at(self, t: datetime) -> Optional[tuple[float, float]]:
        """Interpolated (lat, lon) at time t. Returns None if outside data range."""
        epoch = t.timestamp()
        if epoch < self._epochs[0] - 10 or epoch > self._epochs[-1] + 10:
            return None
        lat = float(np.interp(epoch, self._epochs, self._lats))
        lon = float(np.interp(epoch, self._epochs, self._lons))
        return lat, lon

    def get_coords_in_range(
        self, t_start: datetime, t_end: datetime
    ) -> list[tuple[float, float]]:
        """All (lat, lon) recorded between t_start and t_end (inclusive), with
        interpolated endpoints so the trace starts and ends exactly at the bounds."""
        s, e = t_start.timestamp(), t_end.timestamp()
        mask = (self._epochs >= s) & (self._epochs <= e)
        coords = list(zip(self._lats[mask].tolist(), self._lons[mask].tolist()))
        # Prepend interpolated start point
        start_pos = self.get_position_at(t_start)
        if start_pos is not None and (not coords or coords[0] != start_pos):
            coords.insert(0, start_pos)
        # Append interpolated end point
        end_pos = self.get_position_at(t_end)
        if end_pos is not None and (not coords or coords[-1] != end_pos):
            coords.append(end_pos)
        return coords

    def get_traveled_coords(
        self, t: datetime, t_start: Optional[datetime] = None
    ) -> list[tuple[float, float]]:
        """All (lat, lon) from t_start (or timeline start) up to t, plus interpolated endpoint."""
        epoch = t.timestamp()
        s_epoch = t_start.timestamp() if t_start is not None else self._epochs[0]
        mask = (self._epochs >= s_epoch) & (self._epochs <= epoch)
        coords = list(zip(self._lats[mask].tolist(), self._lons[mask].tolist()))
        if t_start is not None:
            start_pos = self.get_position_at(t_start)
            if start_pos is not None and (not coords or coords[0] != start_pos):
                coords.insert(0, start_pos)
        pos = self.get_position_at(t)
        if pos is not None:
            coords.append(pos)
        return coords


@dataclass
class HRZoneConfig:
    """Heart rate zone configuration and color mapping."""
    max_hr: int = 190

    # Zone thresholds as fraction of max HR
    ZONE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    # BGR colors for OpenCV
    ZONE_COLORS = {
        0: (100, 100, 100),   # Below zone 1: dim gray
        1: (128, 128, 128),   # Zone 1: gray
        2: (255, 100, 50),    # Zone 2: blue
        3: (50, 200, 50),     # Zone 3: green
        4: (0, 140, 255),     # Zone 4: orange
        5: (0, 0, 255),       # Zone 5: red
    }

    def get_zone(self, bpm: int) -> int:
        """Return HR zone (1-5) for a given BPM. Returns 0 if below zone 1."""
        fraction = bpm / self.max_hr
        if fraction >= 0.9:
            return 5
        elif fraction >= 0.8:
            return 4
        elif fraction >= 0.7:
            return 3
        elif fraction >= 0.6:
            return 2
        elif fraction >= 0.5:
            return 1
        return 0

    def get_zone_color(self, bpm: int) -> tuple[int, int, int]:
        """Return BGR color for a given BPM based on its zone."""
        return self.ZONE_COLORS[self.get_zone(bpm)]


class HRTimeline:
    """Sorted timeline of HR samples with interpolation support."""

    def __init__(self, samples: list[HRSample]):
        if not samples:
            raise ValueError("HRTimeline requires at least one sample")
        self.samples = sorted(samples, key=lambda s: s.timestamp)
        # Pre-compute epoch arrays for fast interpolation
        self._epochs = np.array([s.timestamp.timestamp() for s in self.samples])
        self._bpms = np.array([s.bpm for s in self.samples], dtype=np.float64)
        self.gps: Optional[GPSTimeline] = None

    @property
    def start_time(self) -> datetime:
        return self.samples[0].timestamp

    @property
    def end_time(self) -> datetime:
        return self.samples[-1].timestamp

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    def get_bpm_at(self, t: datetime) -> Optional[float]:
        """Get interpolated BPM at a given time. Returns None if too far outside data range."""
        epoch = t.timestamp()
        # Allow up to 10 seconds outside the data range
        if epoch < self._epochs[0] - 10 or epoch > self._epochs[-1] + 10:
            return None
        return float(np.interp(epoch, self._epochs, self._bpms))

    def get_window(self, t: datetime, duration_seconds: float = 60.0) -> list[tuple[float, float]]:
        """Get (offset_seconds, bpm) pairs for the window [t - duration, t].

        offset_seconds is relative to the window start (0 = oldest, duration = current time).
        Returns raw data points that fall within the window.
        """
        t_epoch = t.timestamp()
        window_start = t_epoch - duration_seconds

        points = []
        for epoch, bpm in zip(self._epochs, self._bpms):
            if epoch < window_start:
                continue
            if epoch > t_epoch:
                break
            offset = epoch - window_start
            points.append((offset, float(bpm)))

        # Add interpolated point at current time if we have data
        current_bpm = self.get_bpm_at(t)
        if current_bpm is not None:
            points.append((duration_seconds, current_bpm))

        return points
