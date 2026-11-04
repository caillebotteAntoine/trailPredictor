import math

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.speed_columns import add_speed_columns, format_pace


def test_add_speed_columns_matches_numeric_definitions():
    values = np.array([0.0, np.nan, 1e-12, 3.2, 1e12, -1.0])
    result = add_speed_columns(pd.DataFrame({"value": values}), "value", "speed")

    expected_speed = np.where(values >= 0, values, np.nan)
    expected_speed[1] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_pace = 1000.0 / (60.0 * expected_speed)
        expected_log_speed = np.log(expected_speed)
        expected_log_pace = np.log(expected_pace)

    np.testing.assert_allclose(result["speed"], expected_speed, equal_nan=True)
    np.testing.assert_allclose(result["pace"], expected_pace, equal_nan=True)
    np.testing.assert_allclose(result["log_speed"], expected_log_speed, equal_nan=True)
    np.testing.assert_allclose(result["log_pace"], expected_log_pace, equal_nan=True)
    assert all(pd.api.types.is_float_dtype(result[col]) for col in result.columns[-4:])


@pytest.mark.parametrize(
    ("unit", "value", "expected_speed"),
    [("pace", 5.0, 1000 / 300), ("km_per_h", 36.0, 10.0)],
)
def test_add_speed_columns_accepts_source_units(unit, value, expected_speed):
    result = add_speed_columns(pd.DataFrame({"value": [value]}), "value", unit)
    assert result.loc[0, "speed"] == pytest.approx(expected_speed)


def test_add_speed_columns_does_not_mutate_input():
    source = pd.DataFrame({"value": [1.0]})
    result = add_speed_columns(source, "value", "speed")
    assert list(source.columns) == ["value"]
    assert result is not source


def test_format_pace_matches_speed_formatting():
    assert format_pace(5.0) == "5:00 min/km"
    assert format_pace(5 + 59 / 60) == "5:59 min/km"
    assert format_pace(0.0) == "0:00 min/km"
    assert format_pace(math.nan) == "--:-- min/km"
    assert format_pace(math.inf) == "--:-- min/km"
