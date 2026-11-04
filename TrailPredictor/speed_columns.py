"""Vectorized numeric speed and pace columns."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


SourceUnit = Literal["speed", "pace", "km_per_h"]
SPEED_COLUMNS = ("speed", "pace", "log_speed", "log_pace")
_PACE_FACTOR = 1000.0 / 60.0


def add_speed_columns(
    df: pd.DataFrame,
    from_col: str,
    unit: SourceUnit,
) -> pd.DataFrame:
    """Return a copy with numeric speed, pace, and logarithmic columns.

    ``speed`` is expressed in m/s and ``pace`` in min/km. Missing values are
    propagated. Zero speed produces infinite pace and logarithms. Non-positive
    source pace and negative source speeds are represented as missing values.
    """
    if unit not in ("speed", "pace", "km_per_h"):
        raise ValueError("unit must be 'speed', 'pace', or 'km_per_h'")
    if from_col not in df.columns:
        raise KeyError(f"Source column {from_col!r} not found")

    result = df.copy()
    source = pd.to_numeric(result[from_col], errors="coerce").astype("float64")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if unit == "speed":
            speed = source.where(source >= 0)
        elif unit == "km_per_h":
            speed = (source / 3.6).where(source >= 0)
        else:
            speed = (_PACE_FACTOR / source).where(source > 0)

        pace = _PACE_FACTOR / speed
        log_speed = np.log(speed)
        log_pace = np.log(pace)

    result["speed"] = speed.astype("float64")
    result["pace"] = pace.astype("float64")
    result["log_speed"] = log_speed.astype("float64")
    result["log_pace"] = log_pace.astype("float64")
    return result


def format_pace(pace_min_per_km: float) -> str:
    """Format a numeric pace as ``m:ss min/km``."""
    pace = float(pace_min_per_km)
    if not np.isfinite(pace) or pace < 0:
        return "--:-- min/km"
    total_seconds = round(pace * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d} min/km"
