"""Tests for the pause markers of :class:`TrailPredictor.plot.PlotActivities`."""

import numpy as np
import pytest

from TrailPredictor.activities import RunningActivities
from TrailPredictor.plot import PlotActivities

N_SEGMENTS = 5
SEGMENT_WIDTH = 100.0  # m
# 1 % slope: low running cost, so only the speed decides the pause
FLAT_ALTITUDE = [100.0, 101.0, 102.0, 103.0, 104.0]


@pytest.fixture
def make_plot(make_segments):
    """Return a builder of ``PlotActivities`` with the given paused segments."""

    def _make_plot(paused: list[bool]) -> PlotActivities:
        segments = make_segments(
            FLAT_ALTITUDE, segment_width=SEGMENT_WIDTH, paused=paused
        )
        return PlotActivities(RunningActivities(segments))

    return _make_plot


def _pause_traces(fig) -> list:
    return [trace for trace in fig.data if trace.name == "Pause"]


@pytest.mark.parametrize("method", ["altitude", "pace_by_segment", "hr_overlay"])
def test_pause_markers_on_paused_segments(make_plot, method: str) -> None:
    plot = make_plot([False, True, False, True, False])

    traces = _pause_traces(getattr(plot, method)())

    assert len(traces) == 1
    np.testing.assert_allclose(traces[0].x, [0.2, 0.4])


def test_pause_markers_on_track_map(make_plot) -> None:
    plot = make_plot([True, False, False, False, False])

    traces = _pause_traces(plot.track_map(show_km_markers=False))

    assert len(traces) == 1
    np.testing.assert_allclose(traces[0].lat, [45.0])
    np.testing.assert_allclose(traces[0].lon, [6.0])


def test_no_pause_markers_without_pause(make_plot) -> None:
    plot = make_plot([False] * N_SEGMENTS)

    assert _pause_traces(plot.altitude()) == []
