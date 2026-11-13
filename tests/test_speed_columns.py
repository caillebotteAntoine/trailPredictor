"""Tests for :mod:`TrailPredictor.speed_columns`."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.speed_columns import TARGETS, add_speed_columns, format_pace

PACE_FACTOR = 1000.0 / 60.0


@pytest.fixture
def running_speeds() -> pd.DataFrame:
    return pd.DataFrame({"speed": [2.0, 2.5, 2.5, 2.0, 4.0]})


def test_add_speed_columns_adds_all_targets_as_float(running_speeds):
    result = add_speed_columns(running_speeds)

    assert set(TARGETS) <= set(result.columns)
    assert all(pd.api.types.is_float_dtype(result[col]) for col in TARGETS)
    assert len(result) == len(running_speeds)


def test_add_speed_columns_matches_numeric_definitions(running_speeds):
    result = add_speed_columns(running_speeds)

    speed = running_speeds["speed"].to_numpy()
    expected_pace = PACE_FACTOR / speed
    expected_accel_speed = np.array([np.nan, 0.5, 0.0, -0.5, 2.0])
    expected_accel_pace = PACE_FACTOR * expected_accel_speed / speed**2

    np.testing.assert_allclose(result["speed"], speed)
    np.testing.assert_allclose(result["pace"], expected_pace)
    np.testing.assert_allclose(
        result["accel_speed"], expected_accel_speed, equal_nan=True
    )
    np.testing.assert_allclose(
        result["accel_pace"], expected_accel_pace, equal_nan=True
    )
    np.testing.assert_allclose(result["log_speed"], np.log(speed))
    np.testing.assert_allclose(result["log_pace"], np.log(expected_pace))


def test_add_speed_columns_log_of_non_positive_accel_is_nan(running_speeds):
    result = add_speed_columns(running_speeds)

    # first row has no previous value, row 2 is 0.0, row 3 is negative
    expected_log_accel_speed = np.array(
        [np.nan, np.log(0.5), np.nan, np.nan, np.log(2.0)]
    )
    np.testing.assert_allclose(
        result["log_accel_speed"], expected_log_accel_speed, equal_nan=True
    )
    assert result["log_accel_pace"].isna().tolist() == [
        True,
        False,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("bad_speed", [0.0, np.nan, -1.0])
def test_add_speed_columns_non_positive_speed_gives_nan(bad_speed):
    result = add_speed_columns(pd.DataFrame({"speed": [3.0, bad_speed, 3.0]}))
    row = result.loc[1]

    for col in ("pace", "log_speed", "log_pace", "accel_pace", "log_accel_pace"):
        assert math.isnan(row[col]), col
    if bad_speed == 0.0:
        assert row["speed"] == 0.0
    else:
        assert math.isnan(row["speed"])
    assert not np.isinf(result[list(TARGETS)].to_numpy()).any()


def test_add_speed_columns_emits_no_runtime_warning():
    source = pd.DataFrame({"speed": [0.0, np.nan, 1e-12, 3.2, 3.2, 1e12, -1.0]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        add_speed_columns(source)


def test_add_speed_columns_handles_empty_frame():
    result = add_speed_columns(pd.DataFrame({"speed": pd.Series([], dtype="float64")}))

    assert result.empty
    assert set(TARGETS) <= set(result.columns)


def test_add_speed_columns_requires_speed_column():
    with pytest.raises(ValueError, match="speed"):
        add_speed_columns(pd.DataFrame({"pace": [5.0]}))


def test_add_speed_columns_does_not_mutate_input():
    source = pd.DataFrame({"speed": [0.0, np.nan, 2.0]})
    original = source.copy()

    result = add_speed_columns(source)

    pd.testing.assert_frame_equal(source, original)
    assert result is not source


def test_add_speed_columns_keeps_other_columns():
    source = pd.DataFrame({"speed": [2.0, 3.0], "distance": [100.0, 200.0]})

    result = add_speed_columns(source)

    pd.testing.assert_series_equal(result["distance"], source["distance"])


def test_format_pace_matches_speed_formatting():
    assert format_pace(5.0) == "5:00 min/km"
    assert format_pace(5 + 59 / 60) == "5:59 min/km"
    assert format_pace(0.0) == "0:00 min/km"
    assert format_pace(math.nan) == "--:-- min/km"
    assert format_pace(math.inf) == "--:-- min/km"
    assert format_pace(-1.0) == "--:-- min/km"
