"""Tests for :mod:`TrailPredictor.activity_metrics`."""

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.activity_metrics import ActivityMetrics, compute_activity_metrics


@pytest.fixture
def segments() -> pd.DataFrame:
    """Three 1 km segments of 5 minutes: descend 15 m, then climb 5 m."""
    return pd.DataFrame(
        {
            "distance": [1000.0, 2000.0, 3000.0],
            "elapsed_time": pd.to_timedelta([300, 300, 300], unit="s"),
            "altitude": [110.0, 95.0, 100.0],
            "heart_rate": [140.0, 150.0, 160.0],
        }
    )


def test_compute_activity_metrics(segments: pd.DataFrame) -> None:
    metrics = compute_activity_metrics(segments)

    assert metrics == ActivityMetrics(
        distance_km=pytest.approx(3.0),
        duration_s=pytest.approx(900.0),
        elevation_gain_m=pytest.approx(5.0),  # only the climbs count
        average_pace_min_per_km=pytest.approx(5.0),
        average_heart_rate_bpm=pytest.approx(150.0),
    )


def test_elevation_gain_is_positive(segments: pd.DataFrame) -> None:
    descent = segments.assign(altitude=[300.0, 200.0, 100.0])

    assert compute_activity_metrics(descent).elevation_gain_m == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("column", "field"),
    [
        ("altitude", "elevation_gain_m"),
        ("heart_rate", "average_heart_rate_bpm"),
        ("distance", "distance_km"),
    ],
)
def test_missing_column_gives_nan(
    segments: pd.DataFrame, column: str, field: str
) -> None:
    metrics = compute_activity_metrics(segments.drop(columns=column))

    assert np.isnan(getattr(metrics, field))


def test_non_numeric_values_are_ignored(segments: pd.DataFrame) -> None:
    noisy = segments.assign(heart_rate=["140", "n/a", "160"])

    assert compute_activity_metrics(noisy).average_heart_rate_bpm == pytest.approx(
        150.0
    )


def test_zero_distance_gives_nan_pace(segments: pd.DataFrame) -> None:
    metrics = compute_activity_metrics(segments.assign(distance=0.0))

    assert np.isnan(metrics.average_pace_min_per_km)


def test_missing_distance_gives_nan_pace(segments: pd.DataFrame) -> None:
    metrics = compute_activity_metrics(segments.drop(columns="distance"))

    assert np.isnan(metrics.average_pace_min_per_km)


def test_missing_elapsed_time_raises(segments: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        compute_activity_metrics(segments.drop(columns="elapsed_time"))


def test_does_not_modify_input(segments: pd.DataFrame) -> None:
    before = segments.copy()

    compute_activity_metrics(segments)

    pd.testing.assert_frame_equal(segments, before)
