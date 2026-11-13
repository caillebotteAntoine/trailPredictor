"""Vectorized numeric speed and pace columns."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


SourceUnit = Literal["speed", "pace", "accel_pace", "accel_speed"]

Target = Literal[
    "pace",
    "log_pace",
    "speed",
    "log_speed",
    "accel_pace",
    "log_accel_pace",
    "accel_speed",
    "log_accel_speed",
]

TARGETS: tuple[Target, ...] = (
    "pace",
    "log_pace",
    "speed",
    "log_speed",
    "accel_pace",
    "log_accel_pace",
    "accel_speed",
    "log_accel_speed",
)

_PACE_FACTOR = 1000.0 / 60.0


def add_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric speed, pace, and logarithmic columns.

    ``speed`` is expressed in m/s
    ``pace`` in min/km
    ``accel_speed`` in m/s²
    ``accel_pace`` in min/km²

    Missing values are propagated. Zero speed produces infinite pace and
    logarithms. Non-positive source pace and negative source speeds are
    represented as missing values.
    """
    if "speed" not in df.columns:
        raise ValueError(
            f"DataFrame must contain a 'speed' column, got {list(df.columns)}"
        )

    result = df.copy()

    speed = result["speed"].to_numpy(dtype="float64", copy=False)
    pace = _PACE_FACTOR / speed
    accel_speed = result["speed"].diff().to_numpy(dtype="float64", copy=False)
    accel_pace = _PACE_FACTOR * accel_speed / speed**2

    result["speed"] = speed.astype("float64")
    result["pace"] = pace.astype("float64")
    result["accel_speed"] = accel_speed.astype("float64")
    result["accel_pace"] = accel_pace.astype("float64")

    for col in ["speed", "pace", "accel_speed", "accel_pace"]:
        result[col] = result[col].replace([np.nan], 0.0)
        result[f"log_{col}"] = np.log(result[col]).astype("float64")

    return result


def format_pace(pace_min_per_km: float) -> str:
    """Format a numeric pace as ``m:ss min/km``."""
    pace = float(pace_min_per_km)
    if not np.isfinite(pace) or pace < 0:
        return "--:-- min/km"
    total_seconds = round(pace * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d} min/km"
