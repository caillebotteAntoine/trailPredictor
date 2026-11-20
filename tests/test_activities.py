"""Tests for the segment aggregation of ``RunningActivities``."""

import contextlib
import io
import logging
from pathlib import Path

import pandas as pd
import pytest

from TrailPredictor import activities
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


# --- console output (Windows: cp1252 stdout cannot encode emojis)


@pytest.fixture
def cp1252_stdout() -> io.TextIOWrapper:
    """Return a cp1252 stream, as the stdout of a redirected Windows console.

    Tests redirect stdout to it in their body: pytest resets ``sys.stdout``
    between the setup and the call of a test.
    """
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def test_from_computed_file_falls_back_without_printing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cp1252_stdout: io.TextIOWrapper,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing = tmp_path / "missing.csv"
    sentinel = object()
    monkeypatch.setattr(activities, "_get_computed_file_path", lambda *_, **__: missing)
    monkeypatch.setattr(
        RunningActivities, "from_fit_file", classmethod(lambda *_, **__: sentinel)
    )

    with (
        caplog.at_level(logging.INFO, logger=activities.__name__),
        contextlib.redirect_stdout(cp1252_stdout),
    ):
        result = RunningActivities.from_computed_file("activities/1.fit.gz")

    assert result is sentinel
    cp1252_stdout.flush()
    assert not cp1252_stdout.buffer.getvalue()  # type: ignore[attr-defined]
    assert str(missing) in caplog.text
