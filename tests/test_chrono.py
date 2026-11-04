import math

import pandas as pd
import numpy as np

import pytest

from TrailPredictor.chrono import Chrono, Speed, UnsupportedOperationError


def test_speed_construction_and_unit_conversions():
    speed = Speed(36, unit="km_per_h")

    assert isinstance(speed, Speed)
    assert speed.to("m_per_s") == pytest.approx(10)
    assert speed.to("km_per_h") == pytest.approx(36)
    assert speed.to("mph") == pytest.approx(22.3694, rel=1e-4)


def test_speed_pace_conversions_and_formatting():
    speed = Speed.from_pace(5)

    assert speed.to("km_per_h") == pytest.approx(12)
    assert speed.as_pace() == pytest.approx(5)
    assert speed.pace_str() == "5:00 min/km"
    assert str(speed) == "5:00 min/km"
    assert repr(speed) == "Speed(3.33, unit='m_per_s')"

    assert math.isnan(float(Speed.from_pace(None)))
    assert math.isnan(float(Speed.from_pace(pd.NA)))


def test_speed_nan_and_zero_pace():
    assert math.isnan(float(Speed()))
    assert Speed(None, nan_as_zero=True) == 0
    assert math.isnan(float(Speed(None, nan_as_zero=False).as_pace()))
    assert math.isinf(float(Speed(0).as_pace()))
    assert Speed(0).pace_str() == "--:-- min/km"


def test_speed_rejects_invalid_units_and_values():
    with pytest.raises(ValueError, match="unit not supported"):
        Speed(10, unit="invalid")

    with pytest.raises(ValueError, match="unit not supported"):
        Speed(10).to("invalid")

    with pytest.raises(ValueError, match="pace must be positive"):
        Speed.from_pace(0)


def test_speed_add():
    first = Speed(10)
    second = Speed(2)

    assert type(first + second) is Speed
    assert first + second == pytest.approx(12)

    with pytest.raises(TypeError, match="unsupported operand"):
        first + "unsupported"

    assert type(first + 0.5) is Speed
    assert first + 0.5 == pytest.approx(10.5)


def test_speed_sub():
    first = Speed(10)
    second = Speed(2)

    assert type(first - second) is Speed
    assert first - second == pytest.approx(8)

    assert type(first - 0.25) is Speed
    assert first - 0.25 == pytest.approx(9.75)


def test_speed_mul():
    first = Speed(10)
    second = Speed(2)

    with pytest.raises(UnsupportedOperationError, match="unsupported operand"):
        first * "unsupported"

    with pytest.raises(UnsupportedOperationError, match="unsupported operand"):
        first * second

    assert type(first * 2) is Speed
    assert first * 2 == pytest.approx(20)


def test_speed_div():
    first = Speed(10)
    second = Speed(2)

    assert type(first / second) is float
    assert first / second == pytest.approx(5.0)

    with pytest.raises(UnsupportedOperationError, match="unsupported operand"):
        first / "unsupported"

    assert type(first / 2) is Speed
    assert first / 2 == pytest.approx(5)

    with pytest.raises(UnsupportedOperationError, match="unsupported operand"):
        "unsupported" / first


@pytest.mark.parametrize("type_to_test", [list, np.array, pd.Series])
def test_as_speed(type_to_test):
    speeds = type_to_test([5, 10, 15])
    speed_series = Speed.as_speed(speeds)

    expected_type = np.ndarray if type_to_test is np.array else type_to_test
    assert isinstance(speed_series, expected_type)
    print([isinstance(s, Speed) for s in speed_series])
    assert all(isinstance(s, Speed) for s in speed_series)


def test_chrono_construction_and_formatting():
    from_text = Chrono("1:30:00")
    from_speed = Chrono(10, units="m_per_s")
    from_kmh = Chrono(12, units="km_per_h")

    assert from_text == pytest.approx(1.5)
    assert from_speed == pytest.approx(1.6666666667)
    assert from_kmh == pytest.approx(1 / 12)
    assert str(from_text) == "1:30:0"
    assert Chrono.as_chrono("0:45:30") == pytest.approx(0.7583333333)


def test_chrono_rejects_unsupported_values_and_units():
    with pytest.raises(ValueError, match="unknown units"):
        Chrono(10, units="invalid")

    with pytest.raises(ValueError, match="unsupported type"):
        Chrono(None)
