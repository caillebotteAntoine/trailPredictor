"""Tests for :mod:`TrailPredictor.api.activity_service`."""

import json

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.activities import RunningActivities
from TrailPredictor.activity_metrics import ActivityMetrics
from TrailPredictor.api import activity_service
from TrailPredictor.api.activity_service import (
    CHARTS,
    ActivityNotLoadedError,
    ActivityService,
    charts_view,
    metrics_view,
    summarize_activities,
    track_view,
)
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.loading import UnknownActivityError
from TrailPredictor.plot import PlotActivities
from TrailPredictor.plot.track_3d import TRACK_TOOLTIP

ALTITUDE = [100.0, 110.0, 105.0, 120.0, 118.0, 125.0]


@pytest.fixture
def summary() -> pd.DataFrame:
    """Strava summary: three runs (one without date) and a ride."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "type": ["run", "run", "ride", "run"],
            "name": ["Short run", "Long run", "Ride", None],
            "date": ["2 janv. 2026, 08:00:00", "3 févr. 2026, 07:30:00", "", "?"],
            "distance_m": [5000.0, 42195.0, 60000.0, np.nan],
            "file_name": ["a/1.fit.gz", "a/2.fit.gz", "a/3.fit", "a/4.fit"],
        }
    )


@pytest.fixture
def plot(make_segments) -> PlotActivities:
    return PlotActivities(RunningActivities(make_segments(ALTITUDE)))


@pytest.fixture
def loaded_files() -> list[tuple[str, int]]:
    """Files read by the fake activity loader of `service`."""
    return []


@pytest.fixture
def service(summary: pd.DataFrame, make_segments, loaded_files) -> ActivityService:
    def load(file_name: str, width: int, progress: ProgressCallback):
        loaded_files.append((file_name, width))
        progress(0.5, "halfway")
        return RunningActivities(make_segments(ALTITUDE))

    return ActivityService(lambda: summary, load, max_loaded=2)


def _no_progress(_fraction: float, _message: str) -> None:
    pass


# --- summarize_activities


def test_summarize_activities(summary: pd.DataFrame) -> None:
    rows = summarize_activities(summary)

    assert [row.id for row in rows] == [1, 2, 3, 4]
    assert rows[1].name == "Long run"
    assert rows[1].date == "2026-02-03T07:30:00"
    assert rows[1].distance_km == pytest.approx(42.195)


def test_summarize_activities_missing_values(summary: pd.DataFrame) -> None:
    last = summarize_activities(summary)[-1]

    assert last.name == ""
    assert last.date is None
    assert last.distance_km is None


def test_summarize_activities_is_json_serializable(summary: pd.DataFrame) -> None:
    rows = [row.model_dump() for row in summarize_activities(summary)]

    assert json.loads(json.dumps(rows, allow_nan=False)) == rows


def test_summarize_activities_empty(summary: pd.DataFrame) -> None:
    assert summarize_activities(summary.iloc[0:0]) == []


# --- metrics_view


def test_metrics_view_replaces_nan_with_none() -> None:
    metrics = ActivityMetrics(1.0, 2.0, np.nan, np.inf, 140.0)

    assert metrics_view(metrics) == {
        "distance_km": 1.0,
        "duration_s": 2.0,
        "elevation_gain_m": None,
        "average_pace_min_per_km": None,
        "average_heart_rate_bpm": 140.0,
    }


# --- charts_view


def test_charts_view(plot: PlotActivities) -> None:
    view = charts_view(plot, x_axis="time")

    assert set(view.charts) == set(CHARTS)
    assert all(chart.error is None for chart in view.charts.values())
    assert all({"data", "layout"} <= set(c.figure) for c in view.charts.values())
    assert view.metrics["distance_km"] == pytest.approx(6.0)
    assert view.metrics["elevation_gain_m"] == pytest.approx(10.0 + 15.0 + 7.0)
    json.dumps(view.model_dump(), allow_nan=False)  # valid JSON


def test_charts_view_reports_a_failing_chart(
    plot: PlotActivities, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_plot, _x_axis):
        raise ValueError("no heart rate")

    monkeypatch.setitem(activity_service.CHARTS, "heart_rate", broken)

    view = charts_view(plot)

    assert view.charts["heart_rate"].figure is None
    assert view.charts["heart_rate"].error == "no heart rate"
    assert view.charts["altitude"].figure is not None


# --- track_view


@pytest.mark.parametrize("color_by", ["altitude", "heart_rate", "pace"])
def test_track_view(plot: PlotActivities, color_by: str) -> None:
    view = track_view(plot, color_by=color_by, texture="satellite")  # type: ignore[arg-type]

    assert view.deck["layers"]
    assert "initialViewState" in view.deck
    assert view.tooltip == TRACK_TOOLTIP
    assert "Esri" in view.legend.attribution
    json.dumps(view.model_dump(), allow_nan=False)


def test_track_view_without_gps_raises(make_segments) -> None:
    no_gps = make_segments(ALTITUDE).assign(position_lat=np.nan, position_long=np.nan)
    plot = PlotActivities(RunningActivities(no_gps))

    with pytest.raises(ValueError):
        track_view(plot)


# --- ActivityService


def test_running_activities_are_read_once(summary: pd.DataFrame) -> None:
    calls = []

    def read() -> pd.DataFrame:
        calls.append(1)
        return summary

    service = ActivityService(read)
    service.running_activities()
    runs = service.running_activities()

    assert runs["id"].tolist() == [1, 2, 4]
    assert len(calls) == 1


def test_missing_summary_is_retried(summary: pd.DataFrame) -> None:
    files = [FileNotFoundError("activities.csv"), summary]

    def read() -> pd.DataFrame:
        result = files.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    service = ActivityService(read)

    with pytest.raises(FileNotFoundError):
        service.running_activities()
    assert len(service.running_activities()) == 3


def test_summaries_are_sorted_longest_first(service: ActivityService) -> None:
    assert [row.id for row in service.summaries()] == [2, 1, 4]


def test_file_name(service: ActivityService) -> None:
    assert service.file_name(2) == "a/2.fit.gz"


@pytest.mark.parametrize("activity_id", [3, 99], ids=["not_a_run", "missing"])
def test_file_name_unknown_raises(service: ActivityService, activity_id: int) -> None:
    with pytest.raises(UnknownActivityError):
        service.file_name(activity_id)


def test_load_caches_the_activity(
    service: ActivityService, loaded_files: list[tuple[str, int]]
) -> None:
    reports: list[tuple[float, str]] = []

    service.load(1, 100, lambda fraction, text: reports.append((fraction, text)))
    service.load(1, 100, _no_progress)

    assert loaded_files == [("a/1.fit.gz", 100)]
    assert reports == [(0.5, "halfway")]
    assert isinstance(service.plot(1, 100, 2.0), PlotActivities)


@pytest.mark.parametrize(
    ("activity_id", "name"), [(2, "Long run"), (4, "")], ids=["named", "missing"]
)
def test_load_names_the_activity_from_the_summary(
    service: ActivityService, activity_id: int, name: str
) -> None:
    service.load(activity_id, 100, _no_progress)

    plot = service.plot(activity_id, 100, 2.0)

    assert plot.name == name
    assert plot.altitude().layout.title.subtitle.text == name


def test_load_evicts_the_least_recently_used(
    service: ActivityService, loaded_files: list[tuple[str, int]]
) -> None:
    service.load(1, 100, _no_progress)
    service.load(2, 100, _no_progress)
    service.plot(1, 100, 2.0)  # 1 becomes the most recently used
    service.load(4, 100, _no_progress)

    service.plot(1, 100, 2.0)
    service.plot(4, 100, 2.0)
    with pytest.raises(ActivityNotLoadedError):
        service.plot(2, 100, 2.0)
    assert len(loaded_files) == 3


def test_load_unknown_activity_raises(service: ActivityService) -> None:
    with pytest.raises(UnknownActivityError):
        service.load(99, 100, _no_progress)


def test_plot_of_a_loaded_activity(service: ActivityService) -> None:
    service.load(1, 100, _no_progress)

    plot = service.plot(1, 100, drop_threshold=8.0)

    assert isinstance(plot, PlotActivities)
    assert set(plot.data["drop_category"]) <= {"Uphill", "Downhill", "Flat"}


def test_plot_does_not_modify_the_cached_activity(service: ActivityService) -> None:
    service.load(1, 100, _no_progress)
    first = service.plot(1, 100, drop_threshold=2.0).data

    service.plot(1, 100, drop_threshold=8.0)

    pd.testing.assert_frame_equal(service.plot(1, 100, drop_threshold=2.0).data, first)


def test_plot_of_an_activity_not_loaded_raises(service: ActivityService) -> None:
    service.load(1, 100, _no_progress)

    with pytest.raises(ActivityNotLoadedError):
        service.plot(1, 200, 2.0)
