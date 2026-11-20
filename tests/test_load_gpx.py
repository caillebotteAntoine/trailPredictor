"""Tests for :mod:`TrailPredictor.load_gpx`."""

import gzip
from collections.abc import Callable, Sequence
from pathlib import Path

import gpxpy.geo
import gpxpy.gpx
import numpy as np
import pytest

from TrailPredictor.load_gpx import (
    TRACK_POINT_COLUMNS,
    EmptyGpxError,
    gpx_track_points,
    haversine_distance,
    load_gpx_file,
)

ROUTE_NAME = "Test route"
START_LAT = 45.0  # degrees
START_LONG = 4.0  # degrees
LAT_STEP = 0.001  # degrees, about 111.2 m along a meridian
METRES_PER_LAT_STEP = gpxpy.geo.EARTH_RADIUS * np.radians(LAT_STEP)
DISTANCE_RTOL = 1e-3  # gpxpy approximates the Earth locally

GpxFactory = Callable[..., gpxpy.gpx.GPX]


def _make_gpx(elevations: Sequence[float | None], n_segments: int = 1) -> gpxpy.gpx.GPX:
    """Build a northbound route split evenly into `n_segments` track segments."""
    gpx = gpxpy.gpx.GPX()
    gpx.name = ROUTE_NAME
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    points = [
        gpxpy.gpx.GPXTrackPoint(START_LAT + i * LAT_STEP, START_LONG, elevation=ele)
        for i, ele in enumerate(elevations)
    ]
    for chunk in np.array_split(np.arange(len(points)), n_segments):
        segment = gpxpy.gpx.GPXTrackSegment([points[i] for i in chunk])
        track.segments.append(segment)
    return gpx


@pytest.fixture
def make_gpx() -> GpxFactory:
    return _make_gpx


@pytest.mark.parametrize("compressed", [False, True])
def test_load_gpx_file_reads_plain_and_gzip(
    make_gpx: GpxFactory, tmp_path: Path, compressed: bool
) -> None:
    xml = make_gpx([100.0, 110.0, 105.0]).to_xml().encode("utf-8")
    path = tmp_path / ("route.gpx.gz" if compressed else "route.gpx")
    path.write_bytes(gzip.compress(xml) if compressed else xml)

    gpx = load_gpx_file(path)

    assert gpx.name == ROUTE_NAME
    assert gpx.get_track_points_no() == 3


def test_load_gpx_file_reports_progress(make_gpx: GpxFactory, tmp_path: Path) -> None:
    path = tmp_path / "route.gpx"
    path.write_text(make_gpx([100.0, 110.0]).to_xml(), encoding="utf-8")
    fractions: list[float] = []

    load_gpx_file(path, progress_cb=lambda fraction, _text: fractions.append(fraction))

    assert fractions[0] == pytest.approx(0.0)
    assert fractions[-1] == pytest.approx(1.0)


def test_gpx_track_points_columns_and_distance(make_gpx: GpxFactory) -> None:
    points = gpx_track_points(make_gpx([100.0, 110.0, 105.0]))

    assert points.columns.tolist() == TRACK_POINT_COLUMNS
    assert (points.dtypes == np.float64).all()
    np.testing.assert_allclose(
        points["distance"],
        [0.0, METRES_PER_LAT_STEP, 2 * METRES_PER_LAT_STEP],
        rtol=DISTANCE_RTOL,
    )
    np.testing.assert_allclose(points["altitude"], [100.0, 110.0, 105.0])
    np.testing.assert_allclose(points["position_long"], START_LONG)


def test_gpx_track_points_chains_track_segments(make_gpx: GpxFactory) -> None:
    points = gpx_track_points(make_gpx([100.0] * 4, n_segments=2))

    assert points["distance"].is_monotonic_increasing
    assert points["distance"].iloc[-1] == pytest.approx(
        3 * METRES_PER_LAT_STEP, rel=DISTANCE_RTOL
    )


def test_gpx_track_points_drops_points_without_elevation(
    make_gpx: GpxFactory,
) -> None:
    points = gpx_track_points(make_gpx([100.0, None, 120.0]))

    assert points["altitude"].tolist() == pytest.approx([100.0, 120.0])
    # the dropped point still counts in the distance
    assert points["distance"].iloc[-1] == pytest.approx(
        2 * METRES_PER_LAT_STEP, rel=DISTANCE_RTOL
    )
    assert points.index.tolist() == [0, 1]


@pytest.mark.parametrize("elevations", [[], [None, None]])
def test_gpx_track_points_without_elevation_raises(
    make_gpx: GpxFactory, elevations: list[float | None]
) -> None:
    with pytest.raises(EmptyGpxError):
        gpx_track_points(make_gpx(elevations))


@pytest.mark.parametrize(
    ("lat", "long", "expected"),
    [
        ([], [], []),
        ([45.0], [4.0], [0.0]),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 111_319.49, 111_319.49]),
    ],
)
def test_haversine_distance(
    lat: list[float], long: list[float], expected: list[float]
) -> None:
    np.testing.assert_allclose(haversine_distance(lat, long), expected, rtol=1e-6)
