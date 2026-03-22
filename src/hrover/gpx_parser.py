"""Parse Garmin GPX files to extract heart rate data."""

import xml.etree.ElementTree as ET
from pathlib import Path

import gpxpy

from .hr_data import GPSPoint, GPSTimeline, HRSample, HRTimeline

# Known Garmin TrackPointExtension namespaces
GARMIN_NS_V1 = "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}"
GARMIN_NS_V2 = "{http://www.garmin.com/xmlschemas/TrackPointExtension/v2}"


def _find_hr_in_extensions(extensions: list) -> int | None:
    """Extract heart rate value from GPX trackpoint extensions."""
    for ext in extensions:
        # Try v1 and v2 namespaces
        for ns in (GARMIN_NS_V1, GARMIN_NS_V2):
            # Direct <hr> element
            hr_elem = ext.find(f"{ns}hr")
            if hr_elem is not None and hr_elem.text:
                return int(hr_elem.text)
            # Nested under <TrackPointExtension>
            tpe = ext.find(f"{ns}TrackPointExtension")
            if tpe is not None:
                hr_elem = tpe.find(f"{ns}hr")
                if hr_elem is not None and hr_elem.text:
                    return int(hr_elem.text)
        # Fallback: search any child named "hr" regardless of namespace
        for child in ext.iter():
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "hr" and child.text:
                return int(child.text)
    return None


def parse_gpx(filepath: str | Path) -> tuple[list[HRSample], list[GPSPoint]]:
    """Parse a GPX file and return HR samples and GPS points.

    Raises ValueError if no heart rate data is found.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    hr_samples = []
    gps_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    continue
                hr = _find_hr_in_extensions(point.extensions)
                if hr is not None:
                    hr_samples.append(HRSample(timestamp=point.time, bpm=hr))
                if point.latitude is not None and point.longitude is not None:
                    gps_points.append(GPSPoint(
                        timestamp=point.time,
                        lat=point.latitude,
                        lon=point.longitude,
                    ))

    if not hr_samples:
        raise ValueError(
            f"No heart rate data found in {filepath}. "
            "Ensure the GPX file contains Garmin TrackPointExtension HR data."
        )

    return hr_samples, gps_points


def parse_gpx_to_timeline(filepath: str | Path) -> HRTimeline:
    """Parse a GPX file and return an HRTimeline ready for interpolation.

    If GPS coordinates are present, attaches a GPSTimeline to hr_timeline.gps.
    """
    hr_samples, gps_points = parse_gpx(filepath)
    timeline = HRTimeline(hr_samples)
    if len(gps_points) >= 2:
        timeline.gps = GPSTimeline(gps_points)
    return timeline
