"""Shared pytest fixtures: factories building the test activities.

The factories are exposed as fixtures returning a builder function, so each
test chooses the values it needs while the column layout stays in one place.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.modeling.config import ModelConfig
from TrailPredictor.speed_columns import add_speed_columns

RUNNING_SPEED = 3.0  # m/s
PAUSE_SPEED = 0.5  # m/s, below the pause speed threshold of build_features
DEFAULT_HEART_RATE = 140.0  # bpm
START_TIME = pd.Timestamp("2026-01-01")
START_LAT = 45.0  # degrees
START_LONG = 6.0  # degrees
LAT_LONG_STEP = 0.0025  # degrees between two consecutive segments
ACTIVITY_NAME = "Test run"
N_SPLITS = 2  # the test dataset holds two activities
TEST_SIZE = 0.2  # 20 % test size: one test activity
N_ACTIVITIES_SPLIT = 10  # 20 % test size: two test activities
TWO_ZONES = {"low": (0, 160), "high": (160, 250)}
CUSTOM_DROP_THRESHOLD = 8.0  # m: the -5 m drops become "Flat"
SEED = 0
LASSO_N_ACTIVITIES = 6
LASSO_N_SEGMENTS = 8
MAX_STEP_M = 15  # altitude change between two segments, in metres
SLOPE_SPEED_FACTOR = 0.05  # m/s lost per metre climbed in a segment

SegmentsFactory = Callable[..., pd.DataFrame]


def _make_segments(
    altitude: Sequence[float],
    *,
    segment_width: float = 1000.0,
    speed: float | Sequence[float] = RUNNING_SPEED,
    paused: Sequence[bool] | None = None,
    heart_rate: float | Sequence[float] = DEFAULT_HEART_RATE,
) -> pd.DataFrame:
    """Build one activity already split into segments.

    The columns mirror the output of ``RunningActivities.from_fit_record``.

    Parameters
    ----------
    altitude : sequence of float
        Mean altitude of each segment, in metres; sets the number of segments.
    segment_width : float, optional
        Segment length, in metres. Default is 1000.0.
    speed : float or sequence of float, optional
        Speed of each segment, in m/s. Default is ``RUNNING_SPEED``.
    paused : sequence of bool, optional
        Segments forced to ``PAUSE_SPEED``; the others keep `speed`.
    heart_rate : float or sequence of float, optional
        Heart rate of each segment, in bpm. Default is ``DEFAULT_HEART_RATE``.

    Returns
    -------
    pandas.DataFrame
        One row per segment, with cumulative ``distance`` and the speed /
        pace columns of ``add_speed_columns``.
    """
    n_segments = len(altitude)
    altitude_m = np.asarray(altitude, dtype=np.float64)
    speed_ms = np.broadcast_to(np.asarray(speed, dtype=np.float64), (n_segments,))
    if paused is not None:
        speed_ms = np.where(np.asarray(paused, dtype=bool), PAUSE_SPEED, speed_ms)
    duration_s = segment_width / speed_ms
    start_s = np.concatenate(([0.0], np.cumsum(duration_s)[:-1]))
    altitude_change = np.diff(altitude_m, prepend=altitude_m[0])
    steps = np.arange(n_segments, dtype=np.float64)

    segments = pd.DataFrame(
        {
            "segment": np.arange(n_segments),
            "timestamp": START_TIME + pd.to_timedelta(start_s, unit="s"),
            "elapsed_time": pd.to_timedelta(duration_s, unit="s"),
            "distance": (steps + 1) * segment_width,
            "heart_rate": np.broadcast_to(
                np.asarray(heart_rate, dtype=np.float64), (n_segments,)
            ),
            "altitude": altitude_m,
            "position_lat": START_LAT + steps * LAT_LONG_STEP,
            "position_long": START_LONG + steps * LAT_LONG_STEP,
            "drop_pos": altitude_change.clip(min=0.0),
            "drop_neg": altitude_change.clip(max=0.0),
            "speed": speed_ms,
            "name": ACTIVITY_NAME,
        }
    )
    return add_speed_columns(segments)


@pytest.fixture
def make_segments() -> SegmentsFactory:
    """Return the builder of a segmented activity (see ``_make_segments``)."""
    return _make_segments


@pytest.fixture
def make_dataset() -> Callable[[dict[int, pd.DataFrame]], pd.DataFrame]:
    """Return a builder stacking activities into one dataset.

    The builder takes ``{activity_id: segments}`` and returns the segments of
    all activities with an ``activity_id`` column.
    """

    def _make_dataset(activities: dict[int, pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(
            [
                segments.assign(activity_id=activity_id)
                for activity_id, segments in activities.items()
            ],
            ignore_index=True,
        )

    return _make_dataset


@pytest.fixture
def make_fit_record() -> SegmentsFactory:
    """Return a builder of raw FIT ``record`` messages.

    The builder takes the cumulative ``distance`` and the ``altitude`` of
    each record (in metres) and an optional ``period_s`` between records
    (default 10 s).
    """

    def _make_fit_record(
        distance: Sequence[float], altitude: Sequence[float], period_s: int = 10
    ) -> pd.DataFrame:
        n_records = len(distance)
        altitude_m = np.asarray(altitude, dtype=np.float64)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    START_TIME, periods=n_records, freq=f"{period_s}s"
                ),
                "distance": np.asarray(distance, dtype=np.float64),
                "altitude": altitude_m,
                "enhanced_altitude": altitude_m,
                "position_lat": np.zeros(n_records, dtype=np.int64),
                "position_long": np.zeros(n_records, dtype=np.int64),
                "heart_rate": np.full(n_records, DEFAULT_HEART_RATE),
            }
        )

    return _make_fit_record


@pytest.fixture
def dataset(make_segments: SegmentsFactory, make_dataset) -> pd.DataFrame:
    """Two activities of 1 km segments, with hand-checked features.

    Activity 1: drops [0, 10, -5, 10, 0, -15], pause on the 3rd segment.
    Activity 2: drops [0, 0, 10, -5], pause on the last segment.
    """
    return make_dataset(
        {
            1: make_segments(
                [100, 110, 105, 115, 115, 100],
                paused=[False, False, True, False, False, False],
                heart_rate=[100.0, 150, 165, 175, 185, 195],
            ),
            2: make_segments(
                [200, 200, 210, 205],
                paused=[False, False, False, True],
                heart_rate=[142.0, 141, 220, 130],
            ),
        }
    )


@pytest.fixture
def lasso_data(make_segments: SegmentsFactory, make_dataset) -> pd.DataFrame:
    """Activities whose speed decreases with the climb of each segment."""
    rng = np.random.default_rng(SEED)
    activities = {}
    for activity_id in range(LASSO_N_ACTIVITIES):  # few activities: loop over inputs
        steps = rng.integers(-MAX_STEP_M, MAX_STEP_M + 1, size=LASSO_N_SEGMENTS)
        steps[0] = 0
        # climb and descend in every activity: a zero total climb or descent
        # makes the done / remaining ratios 0 / 0 = NaN in build_features
        steps[1:3] = [MAX_STEP_M, -MAX_STEP_M]
        altitude = 100.0 + np.cumsum(steps)
        speed = RUNNING_SPEED - SLOPE_SPEED_FACTOR * steps
        activities[activity_id] = make_segments(altitude, speed=speed)
    return make_dataset(activities)


@pytest.fixture
def model_config() -> ModelConfig:
    """Return a default model config for testing."""
    return ModelConfig(seed=SEED, n_splits=N_SPLITS, test_size=TEST_SIZE)
