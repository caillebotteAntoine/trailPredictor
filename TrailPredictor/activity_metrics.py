"""Summary metrics of a segmented running activity."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

METRES_PER_KM = 1000.0
SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class ActivityMetrics:
    """Summary metrics of an activity; NaN when the data is missing.

    Parameters
    ----------
    distance_km : float
        Total distance, in km.
    duration_s : float
        Total elapsed time, in seconds.
    elevation_gain_m : float
        Positive elevation gain, in metres.
    average_pace_min_per_km : float
        Average pace over the elapsed time, in min/km.
    average_heart_rate_bpm : float
        Mean heart rate, in bpm.
    """

    distance_km: float
    duration_s: float
    elevation_gain_m: float
    average_pace_min_per_km: float
    average_heart_rate_bpm: float


def _numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return `column` as floats, or an all-NaN series if it is missing."""
    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype=np.float64)
    return pd.to_numeric(data[column], errors="coerce").astype(np.float64)


def compute_activity_metrics(data: pd.DataFrame) -> ActivityMetrics:
    """Compute the summary metrics of an activity.

    Parameters
    ----------
    data : pandas.DataFrame
        Segments of the activity, in order, with an ``elapsed_time``
        timedelta column and optional ``distance`` (cumulative, in metres),
        ``altitude`` (metres) and ``heart_rate`` (bpm) columns.

    Returns
    -------
    ActivityMetrics
        Metrics of the activity. The elevation gain is the sum of the positive
        altitude changes between consecutive segments; the pace is NaN when
        the distance is not positive.

    Raises
    ------
    KeyError
        If `data` has no ``elapsed_time`` column.
    """
    distance_km = _numeric_column(data, "distance").max() / METRES_PER_KM
    duration_s = float(data["elapsed_time"].dt.total_seconds().sum())
    altitude = _numeric_column(data, "altitude")
    elevation_gain_m = (
        float(altitude.diff().clip(lower=0).sum()) if altitude.notna().any() else np.nan
    )
    average_pace = (
        duration_s / SECONDS_PER_MINUTE / distance_km if distance_km > 0 else np.nan
    )
    return ActivityMetrics(
        distance_km=float(distance_km),
        duration_s=duration_s,
        elevation_gain_m=elevation_gain_m,
        average_pace_min_per_km=float(average_pace),
        average_heart_rate_bpm=float(_numeric_column(data, "heart_rate").mean()),
    )
