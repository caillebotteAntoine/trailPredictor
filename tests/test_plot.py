"""Tests for the pause markers of :class:`TrailPredictor.plot.PlotActivities`."""

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.activities import RunningActivities
from TrailPredictor.plot import PlotActivities
from TrailPredictor.speed_columns import TARGETS

N_SEGMENTS = 5


def _activity_data(pause: list[bool] | None) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=N_SEGMENTS, freq="60s"),
            "elapsed_time": pd.to_timedelta(np.arange(N_SEGMENTS) * 60, unit="s"),
            "position_lat": np.linspace(45.0, 45.01, N_SEGMENTS),
            "position_long": np.linspace(6.0, 6.01, N_SEGMENTS),
            "distance": np.arange(1, N_SEGMENTS + 1) * 100.0,
            "altitude": [100.0, 110.0, 105.0, 120.0, 115.0],
            "heart_rate": [140.0, 150.0, 160.0, 155.0, 145.0],
            "name": "Test run",
        }
    )
    for target in TARGETS:
        data[target] = np.linspace(5.0, 6.0, N_SEGMENTS)
    if pause is not None:
        data["pause"] = pause
    return data


def _pause_traces(fig) -> list:
    return [trace for trace in fig.data if trace.name == "Pause"]


@pytest.mark.parametrize("method", ["altitude", "pace_by_segment", "hr_overlay"])
def test_pause_markers_on_paused_segments(method: str) -> None:
    pause = [False, True, False, True, False]
    plot = PlotActivities(RunningActivities(_activity_data(pause)))

    traces = _pause_traces(getattr(plot, method)())

    assert len(traces) == 1
    np.testing.assert_allclose(traces[0].x, [0.2, 0.4])


def test_pause_markers_on_track_map() -> None:
    pause = [True, False, False, False, False]
    plot = PlotActivities(RunningActivities(_activity_data(pause)))

    traces = _pause_traces(plot.track_map(show_km_markers=False))

    assert len(traces) == 1
    np.testing.assert_allclose(traces[0].lat, [45.0])
    np.testing.assert_allclose(traces[0].lon, [6.0])


@pytest.mark.parametrize(
    "pause", [None, [False] * N_SEGMENTS], ids=["missing", "all_false"]
)
def test_no_pause_markers_without_pause(pause: list[bool] | None) -> None:
    plot = PlotActivities(RunningActivities(_activity_data(pause)))

    assert _pause_traces(plot.altitude()) == []
