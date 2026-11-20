"""Tests for the segment aggregation of ``RunningActivities``."""

from pathlib import Path

import gpxpy.gpx
import pandas as pd
import pytest

from TrailPredictor import activities as activities_module
from TrailPredictor.activities import RunningActivities

SEGMENT_WIDTH = 100


@pytest.fixture
def fit_record(make_fit_record) -> pd.DataFrame:
    """Six records spread over two 100 m segments."""
    return make_fit_record(
        distance=[0.0, 30.0, 60.0, 110.0, 140.0, 190.0],
        altitude=[100.0, 110.0, 105.0, 120.0, 115.0, 100.0],
    )


def test_from_fit_record_drops_per_segment(fit_record: pd.DataFrame) -> None:
    activity = RunningActivities.from_fit_record(fit_record, SEGMENT_WIDTH)
    data = activity.data

    # segment 0: +10, -5 ; segment 1: +15 (crossing the boundary), -5, -15
    assert data["drop_pos"].tolist() == pytest.approx([10.0, 15.0])
    assert data["drop_neg"].tolist() == pytest.approx([-5.0, -20.0])


def test_from_fit_record_drops_sum_to_altitude_change(
    fit_record: pd.DataFrame,
) -> None:
    activity = RunningActivities.from_fit_record(fit_record, SEGMENT_WIDTH)
    data = activity.data

    net_change = fit_record["altitude"].iloc[-1] - fit_record["altitude"].iloc[0]
    assert data["drop_pos"].sum() + data["drop_neg"].sum() == pytest.approx(net_change)


def test_from_fit_record_does_not_mutate_input(fit_record: pd.DataFrame) -> None:
    original = fit_record.copy()
    RunningActivities.from_fit_record(fit_record, SEGMENT_WIDTH)
    pd.testing.assert_frame_equal(fit_record, original)


# --- planned activity from track points / GPX


@pytest.fixture
def track_points() -> pd.DataFrame:
    """Five track points spread over two 100 m segments."""
    return pd.DataFrame(
        {
            "distance": [0.0, 50.0, 99.0, 150.0, 180.0],
            "altitude": [100.0, 110.0, 105.0, 120.0, 100.0],
            "position_lat": [45.0, 45.1, 45.2, 45.3, 45.4],
            "position_long": [4.0, 4.0, 4.0, 4.0, 4.0],
        }
    )


def test_from_track_points_aggregates_segments(track_points: pd.DataFrame) -> None:
    data = RunningActivities.from_track_points(track_points, SEGMENT_WIDTH).data

    assert data["segment"].tolist() == [0, 1]
    assert data["distance"].tolist() == [100, 200]
    assert data["altitude"].tolist() == pytest.approx([105.0, 110.0])
    assert data["position_lat"].tolist() == pytest.approx([45.1, 45.35])
    # segment 0: +10, -5 ; segment 1: +15 (crossing the boundary), -20
    assert data["drop_pos"].tolist() == pytest.approx([10.0, 15.0])
    assert data["drop_neg"].tolist() == pytest.approx([-5.0, -20.0])


def test_from_track_points_leaves_time_and_target_missing(
    track_points: pd.DataFrame,
) -> None:
    data = RunningActivities.from_track_points(track_points, SEGMENT_WIDTH).data

    for col in ["timestamp", "elapsed_time", "heart_rate", "speed", "pace"]:
        assert data[col].isna().all(), col
    assert pd.api.types.is_datetime64_any_dtype(data["timestamp"])
    assert pd.api.types.is_timedelta64_dtype(data["elapsed_time"])


def test_from_track_points_has_fit_record_columns(
    track_points: pd.DataFrame, fit_record: pd.DataFrame
) -> None:
    from_points = RunningActivities.from_track_points(track_points, SEGMENT_WIDTH)
    from_fit = RunningActivities.from_fit_record(fit_record, SEGMENT_WIDTH)

    assert from_points.data.columns.tolist() == from_fit.data.columns.tolist()
    assert from_points.segment_width == SEGMENT_WIDTH


def test_from_track_points_does_not_mutate_input(
    track_points: pd.DataFrame,
) -> None:
    original = track_points.copy()
    RunningActivities.from_track_points(track_points, SEGMENT_WIDTH)
    pd.testing.assert_frame_equal(track_points, original)


@pytest.mark.parametrize(
    ("columns", "message"),
    [(["distance", "altitude"], "miss the columns"), (None, "no track point")],
)
def test_from_track_points_invalid_input_raises(
    track_points: pd.DataFrame, columns: list[str] | None, message: str
) -> None:
    points = track_points[columns] if columns else track_points.iloc[0:0]
    with pytest.raises(ValueError, match=message):
        RunningActivities.from_track_points(points, SEGMENT_WIDTH)


def test_from_gpx_file_names_activity_after_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpx = gpxpy.gpx.GPX()
    gpx.name = "Test route"
    segment = gpxpy.gpx.GPXTrackSegment(
        [
            gpxpy.gpx.GPXTrackPoint(45.0 + 0.001 * i, 4.0, elevation=100.0 + i)
            for i in range(5)  # about 445 m
        ]
    )
    track = gpxpy.gpx.GPXTrack()
    track.segments.append(segment)
    gpx.tracks.append(track)
    (tmp_path / "route.gpx").write_text(gpx.to_xml(), encoding="utf-8")
    monkeypatch.setattr(activities_module, "DATA_DIR", tmp_path)

    activity = RunningActivities.from_gpx_file("route.gpx", SEGMENT_WIDTH)

    assert activity.name == "Test route"
    assert activity.data["distance"].tolist() == [100, 200, 300, 400, 500]
    assert activity.data["drop_pos"].sum() == pytest.approx(4.0)
