"""Tests for the segment aggregation of ``RunningActivities``."""

import pandas as pd
import pytest

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
