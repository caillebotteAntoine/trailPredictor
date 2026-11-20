"""Tests for :mod:`TrailPredictor.modeling.prediction`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, SegmentsFactory

from TrailPredictor.activities import RunningActivities
from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.lasso import LassoConfig
from TrailPredictor.modeling.prediction import predict_activity

ZONE = "Récupération"  # zone of DEFAULT_HEART_RATE in the default zones
ROUTE_ALTITUDE = [100.0, 115.0, 130.0, 130.0, 115.0, 100.0]  # up, flat, down
UPHILL, DOWNHILL = 1, 5  # segments climbing / descending 15 m


@pytest.fixture
def config() -> LassoConfig:
    return LassoConfig(seed=SEED, n_splits=2, inner_n_splits=2, n_alphas=20)


@pytest.fixture
def train(lasso_data: pd.DataFrame) -> Dataset:
    return Dataset(lasso_data)


@pytest.fixture
def route(make_segments: SegmentsFactory) -> RunningActivities:
    """Planned route: no speed, no heart rate."""
    return RunningActivities(
        make_segments(ROUTE_ALTITUDE, speed=np.nan, heart_rate=np.nan)
    )


def _predict(
    train: Dataset, route: RunningActivities, config: LassoConfig, tmp_path: Path
) -> RunningActivities:
    return predict_activity(
        train, route, heart_rate_zone=ZONE, config=config, models_dir=tmp_path
    )


def test_predict_activity_predicts_every_segment(
    train: Dataset, route: RunningActivities, config: LassoConfig, tmp_path: Path
) -> None:
    predicted = _predict(train, route, config, tmp_path)

    assert predicted.target_name == "pace"
    assert predicted.prediction.name == "pace_pred"
    assert len(predicted.prediction) == len(route.data)
    assert np.isfinite(predicted.prediction).all()


def test_predict_activity_is_slower_uphill(
    train: Dataset, route: RunningActivities, config: LassoConfig, tmp_path: Path
) -> None:
    pace = _predict(train, route, config, tmp_path).prediction

    # in lasso_data, the speed decreases with the climb of each segment
    assert pace.iloc[UPHILL] > pace.iloc[DOWNHILL]


def test_predict_activity_leaves_inputs_unchanged(
    train: Dataset, route: RunningActivities, config: LassoConfig, tmp_path: Path
) -> None:
    route_data, train_data = route.data, train.data
    train_features = train.features

    _predict(train, route, config, tmp_path)

    pd.testing.assert_frame_equal(route.data, route_data)
    pd.testing.assert_frame_equal(train.data, train_data)
    assert train.features == train_features


def test_predict_activity_segment_width_mismatch_raises(
    train: Dataset,
    make_segments: SegmentsFactory,
    config: LassoConfig,
    tmp_path: Path,
) -> None:
    route = RunningActivities(make_segments(ROUTE_ALTITUDE, segment_width=500.0))

    with pytest.raises(ValueError, match="Segment width"):
        _predict(train, route, config, tmp_path)


def test_predict_activity_unknown_zone_raises(
    train: Dataset, route: RunningActivities, config: LassoConfig, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="unknown categories"):
        predict_activity(
            train,
            route,
            heart_rate_zone="Not a zone",
            config=config,
            models_dir=tmp_path,
        )
