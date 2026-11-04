"""Chrono and speed utility functions translated from source/chrono.R"""

from __future__ import annotations
from typing import Literal

import math
from typing import Union
from numbers import Real
import numpy as np
import pandas as pd


class UnsupportedOperationError(TypeError):
    """Custom exception for unsupported operations in Speed class."""

    def __init__(self, other):
        message = f"unsupported operand type(s) for operation: 'Speed' and '{type(other).__name__}'"
        super().__init__(message)


SpeedUnit = Literal[
    "m_per_s",
    "log(m_per_s)",
    "km_per_h",
    "log(km_per_h)",
    "mph",
    "log(mph)",
    "min_per_km",
    "log(min_per_km)",
]
SPEED_UNITS: tuple[SpeedUnit, ...] = (
    "m_per_s",
    "log(m_per_s)",
    "km_per_h",
    "log(km_per_h)",
    "mph",
    "log(mph)",
    "min_per_km",
    "log(min_per_km)",
)


class Speed(float):
    """A class representing speed, with support for various units and conversions.
    The speed is stored internally in meters per second (m/s)."""

    # Conversion factors to meters per second (m/s)
    _TO_M_PER_S: dict[SpeedUnit, float] = {
        "m_per_s": 1.0,
        "km_per_h": 1.0 / 3.6,
        "mph": 0.44704,
        "log(m_per_s)": 1.0,
        "log(km_per_h)": 1.0 / 3.6,
        "log(mph)": 0.44704,
    }
    __array_priority__ = (
        1000  # Ensure Speed takes precedence in operations with numpy arrays
    )

    def __new__(
        cls, value=float("nan"), unit: SpeedUnit = "m_per_s", nan_as_zero: bool = False
    ):
        if unit == "min_per_km":
            return cls.from_pace(value)
        if unit == "log(min_per_km)":
            return cls.from_pace(math.exp(value))
        try:
            factor = cls._TO_M_PER_S[unit]
        except KeyError:
            raise ValueError(
                f"unit not supported : {unit!r}. supported units : {sorted(cls._TO_M_PER_S)}"
            ) from None

        if value is None or pd.isna(value):
            value = 0.0 if nan_as_zero else float("nan")

        if unit.startswith("log(") and unit.endswith(")"):
            value = math.exp(float(value))

        ms = float(value) * factor
        return super().__new__(cls, ms)

    # ---------- alternative constructors ----------
    @classmethod
    def from_pace(cls, minutes_per_km: float) -> Speed:
        """Construct a speed from a pace in minutes per kilometer."""
        if minutes_per_km is None or pd.isna(minutes_per_km):
            return cls(float("nan"))
        # if minutes_per_km <= 0:
        #     raise ValueError("pace must be positive")
        return cls(1000.0 / (60.0 * minutes_per_km))

    # ---------- conversions ----------
    def to(self, unit: SpeedUnit = "m_per_s") -> float:
        """Convert the speed to the specified unit."""
        try:
            if unit.startswith("log(") and unit.endswith(")"):
                return math.log(float(self) / self._TO_M_PER_S[unit])
            return float(self) / self._TO_M_PER_S[unit]
        except KeyError:
            raise ValueError(
                f"unit not supported : {unit!r}. supported units : {sorted(self._TO_M_PER_S)}"
            ) from None

    def as_pace(self) -> float:
        """Return the pace in minutes per kilometer.
        Returns NaN if the speed is NaN, and infinity if the speed is zero."""
        v = float(self)
        if math.isnan(v):
            return math.nan
        if v == 0.0:
            return math.inf
        return 1000.0 / (60.0 * v)

    def pace_str(self) -> str:
        """Return the pace as a string in the format 'm:ss min/km'."""
        pace = self.as_pace()
        if not math.isfinite(pace):
            return "--:-- min/km"
        total_s = round(pace * 60)  # to avoid : '5:60'
        m, s = divmod(total_s, 60)
        return f"{m}:{s:02d} min/km"

    # ---------- print ----------
    def __str__(self) -> str:
        return self.pace_str()

    def __repr__(self) -> str:
        return f"Speed({float(self):.2f}, unit='m_per_s')"

    # ---------- arithmetic ----------
    def __add__(self, other):
        if isinstance(other, Real):
            return type(self)(float(self) + float(other))
        raise UnsupportedOperationError(other)

    __radd__ = __add__

    def __sub__(self, other: float) -> float:
        if isinstance(other, Real):
            return type(self)(float(self) - float(other))
        raise UnsupportedOperationError(other)

    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, Speed):
            raise UnsupportedOperationError(other)  # speed * speed as no sense

        if isinstance(other, (pd.Timedelta, np.timedelta64)):
            raise UnsupportedOperationError(other)

        if isinstance(other, Real):
            return type(self)(float(self) * float(other))

        raise UnsupportedOperationError(other)

    def distance(self, duration) -> float:
        """Distance in meters covered during `duration`."""
        if isinstance(duration, (pd.Timedelta, np.timedelta64)):
            seconds = pd.Timedelta(duration).total_seconds()
        else:
            raise TypeError(f"Unknown duration type: {type(duration).__name__}")
        return float(self) * seconds

    @staticmethod
    def total_distance(speeds, durations) -> float:
        """Total distance in meters covered during `durations` at `speeds`."""
        if len(speeds) != len(durations):
            raise ValueError("Speeds and durations must have the same length.")
        total_distance = 0.0
        for speed, duration in zip(speeds, durations):
            total_distance += speed.distance(duration)
        return total_distance

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Speed):
            return float(self) / float(other)  # rate without units
        if isinstance(other, Real):
            return type(self)(float(self) / float(other))
        raise UnsupportedOperationError(other)

    def __rtruediv__(self, other):
        raise UnsupportedOperationError(other)  # other / speed n'a pas de sens

    @staticmethod
    def as_speed(x, unit: str = "m_per_s"):
        if isinstance(x, list):
            return [Speed(v, unit) for v in x]
        if isinstance(x, np.ndarray):
            return np.array([Speed(v, unit) for v in x], dtype=object)
        if isinstance(x, (pd.Series)):
            return pd.Series([Speed(v, unit) for v in x], dtype="object", index=x.index)

        return Speed(x, unit)


class Chrono(float):
    def __new__(cls, value: Union[str, float], units: str = "m_per_s"):
        if isinstance(value, str):
            parts = [int(p) for p in value.split(":")]
            out = parts[0] + (parts[1] + parts[2] / 60.0) / 60.0
            return float.__new__(cls, out)
        if isinstance(value, (int, float)):
            if units == "m_per_s":
                out = 1.0 / (value / 1000.0 * 60.0)
            elif units == "km_per_h":
                out = 1.0 / value
            else:
                raise ValueError("unknown units")
            return float.__new__(cls, out)
        raise ValueError("unsupported type for Chrono")

    def __str__(self):
        h = int(np.floor(self))
        m = int(np.floor((self - h) * 60))
        s = int(np.floor(((self - h) * 60 - m) * 60))
        return f"{h}:{m}:{s}"

    # convenience constructors

    @staticmethod
    def as_chrono(x, units: str = "m_per_s"):
        return Chrono(x, units)
