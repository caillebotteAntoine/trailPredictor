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


def _safe_log(values: np.ndarray) -> np.ndarray:
    """Natural log of strictly positive values; NaN everywhere else."""
    positive = values > 0
    return np.log(values, out=np.full_like(values, np.nan), where=positive)


def add_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric speed, pace, and logarithmic columns.

    Adds ``pace``, ``accel_speed``, ``accel_pace`` and the ``log_*``
    counterparts of all four columns, computed from ``speed``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with a numeric ``speed`` column in m/s. It is not modified.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with ``speed``, ``pace``, ``accel_speed``,
        ``accel_pace`` and their ``log_*`` columns, all as ``float64``.

    Raises
    ------
    ValueError
        If `df` has no ``speed`` column.

    See Also
    --------
    format_pace : Format a pace value for display.

    Notes
    -----
    Units: ``speed`` in m/s, ``pace`` in min/km, ``accel_speed`` in m/s²
    and ``accel_pace`` in min/km².

    Missing values are propagated and negative speeds are treated as
    missing. ``pace`` and ``accel_pace`` are missing where speed is not
    strictly positive. ``log_*`` columns are missing wherever the source
    value is missing or not strictly positive, e.g. the first ``accel_*``
    row or a deceleration.

    Examples
    --------
    >>> result = add_speed_columns(pd.DataFrame({"speed": [2.0, 2.5]}))
    >>> result["pace"].round(2).tolist()
    [8.33, 6.67]
    >>> result["log_accel_speed"].isna().tolist()
    [True, False]
    """
    if "speed" not in df.columns:
        raise ValueError(
            f"DataFrame must contain a 'speed' column, got {list(df.columns)}"
        )

    result = df.copy()

    raw_speed = result["speed"].to_numpy(dtype="float64", copy=True)
    speed = np.where(raw_speed >= 0, raw_speed, np.nan)
    moving = speed > 0
    pace = np.divide(_PACE_FACTOR, speed, out=np.full_like(speed, np.nan), where=moving)
    accel_speed = np.diff(speed, prepend=np.nan)
    accel_pace = np.divide(
        _PACE_FACTOR * accel_speed,
        speed**2,
        out=np.full_like(speed, np.nan),
        where=moving,
    )

    columns = {
        "speed": speed,
        "pace": pace,
        "accel_speed": accel_speed,
        "accel_pace": accel_pace,
    }
    for name, values in columns.items():
        result[name] = values
        result[f"log_{name}"] = _safe_log(values)

    return result


def format_pace(pace_min_per_km: float) -> str:
    """Format a numeric pace as ``m:ss min/km``.

    Seconds are rounded to the nearest integer. Missing, infinite or
    negative paces are shown as a placeholder.

    Parameters
    ----------
    pace_min_per_km : float
        Pace in minutes per kilometre.

    Returns
    -------
    str
        The formatted pace, e.g. ``"5:30 min/km"``, or ``"--:-- min/km"``
        when `pace_min_per_km` is not a finite, non-negative number.

    Examples
    --------
    >>> format_pace(5.5)
    '5:30 min/km'
    >>> format_pace(float("nan"))
    '--:-- min/km'
    """
    pace = float(pace_min_per_km)
    if not np.isfinite(pace) or pace < 0:
        return "--:-- min/km"
    total_seconds = round(pace * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d} min/km"
