"""Tests for :mod:`TrailPredictor.api.heart_rate_service`."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from TrailPredictor.api import heart_rate_service
from TrailPredictor.api.dataset_service import (
    DatasetParams,
    DatasetSelection,
    DatasetService,
)
from TrailPredictor.api.heart_rate_service import (
    MAX_ZONE_COUNT,
    HeartRateService,
    NoCurrentDatasetError,
    UnknownResultError,
    result_view,
    summary_rows,
    trials_chart,
)
from TrailPredictor.dataset import Dataset
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.modeling.heart_rate import (
    MIN_ZONES,
    HeartRateConfig,
    SavedHeartRateResult,
    reference_score,
)

PARAMS = DatasetParams(40, 60, 1000)
OTHER_PARAMS = DatasetParams(10, 30, 1000)
SELECTION = DatasetSelection(
    min_distance=40, max_distance=60, segment_width=1000, target="speed", cuts=[150]
)
N_SPLITS = 2  # the test dataset holds two activities


@pytest.fixture(autouse=True)
def quiet_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def config() -> HeartRateConfig:
    """Small, fast search of 2 and 3 zones."""
    return HeartRateConfig(
        zone_counts=(2, 3), n_trials=2, max_trials_factor=5, n_splits=N_SPLITS
    )


@pytest.fixture
def datasets(dataset: pd.DataFrame) -> DatasetService:
    def load(_min: int, _max: int, _width: int, _progress: ProgressCallback):
        return Dataset(dataset)

    return DatasetService(load, max_loaded=1, saved_results=list)


def _no_progress(_fraction: float, _message: str) -> None:
    pass


@pytest.fixture
def current(datasets: DatasetService) -> DatasetService:
    """Dataset service with a current dataset."""
    datasets.load(PARAMS, _no_progress)
    datasets.select(SELECTION)
    return datasets


@pytest.fixture
def service(datasets: DatasetService, tmp_path: Path) -> HeartRateService:
    return HeartRateService(datasets, models_dir=tmp_path)


def run_at_once(task: Callable[[ProgressCallback], object]) -> str:
    """Stand-in for JobManager.submit: run the task, return a job id."""
    task(_no_progress)
    return "job"


@pytest.fixture
def trials() -> pd.DataFrame:
    """Trials of 2 and 3 zones, as read from the CSV (cuts as strings)."""
    return pd.DataFrame(
        {
            "n_zones": [2, 2, 3, 3, 3],
            "trial_number": [0, 1, 0, 1, 2],
            "rmse": [0.40, 0.30, 0.28, np.nan, 0.32],
            "rmse_se": [0.01, 0.02, 0.03, np.nan, 0.02],
            "cuts": ["[150]", "[140]", "[140, 170]", "[130, 200]", "[150, 175]"],
            "state": ["COMPLETE", "COMPLETE", "COMPLETE", "PRUNED", "COMPLETE"],
        }
    )


@pytest.fixture
def saved(config: HeartRateConfig, tmp_path: Path) -> SavedHeartRateResult:
    return SavedHeartRateResult("pace", config, tmp_path / "x.csv", "key", "named")


# --- context and current dataset


def test_context_without_dataset(service: HeartRateService) -> None:
    context = service.context()

    assert context.dataset is None
    assert context.defaults == HeartRateConfig()
    assert (context.limits.min_zones, context.limits.max_zones) == (
        MIN_ZONES,
        MAX_ZONE_COUNT,
    )


@pytest.mark.usefixtures("current")
def test_context_with_dataset(service: HeartRateService, dataset: pd.DataFrame) -> None:
    context = service.context()

    assert context.dataset is not None
    assert context.dataset.selection == SELECTION
    assert context.dataset.n_rows == len(dataset)


def test_evicted_current_dataset_is_none(
    current: DatasetService, service: HeartRateService
) -> None:
    current.load(OTHER_PARAMS, _no_progress)  # max_loaded=1 evicts PARAMS

    assert service.current_dataset() is None


# --- optimisations


def test_start_optimization_without_dataset_raises(
    service: HeartRateService, config: HeartRateConfig
) -> None:
    with pytest.raises(NoCurrentDatasetError):
        service.start_optimization("speed", config, run_at_once)


@pytest.mark.usefixtures("current")
def test_start_optimization_runs_then_reloads(
    service: HeartRateService, config: HeartRateConfig
) -> None:
    first = service.start_optimization("speed", config, run_at_once)
    second = service.start_optimization("speed", config, run_at_once)

    assert first.job_id == "job"
    assert not first.already_saved
    assert second.already_saved
    assert first.key == second.key
    saved = service.saved_result(first.key)
    assert saved.target_name == "speed"
    assert saved.config == config
    assert [str(s.result_path) for s in service.saved_results()] == [first.key]


def test_saved_result_unknown_key_raises(service: HeartRateService) -> None:
    with pytest.raises(UnknownResultError):
        service.saved_result("C:/Windows/win.ini")


# --- reference


def test_reference_without_dataset_is_none(
    service: HeartRateService, saved: SavedHeartRateResult
) -> None:
    assert service.reference(saved) is None


def test_reference_is_computed_once(
    current: DatasetService,
    service: HeartRateService,
    saved: SavedHeartRateResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def counted(*args):
        calls.append(args)
        return reference_score(*args)

    monkeypatch.setattr(heart_rate_service, "reference_score", counted)
    dataset = current.dataset(PARAMS, "speed", SELECTION.cuts)

    first = service.reference(saved)
    second = service.reference(saved)

    assert first == second
    assert first == pytest.approx(reference_score(dataset, "pace", saved.config))
    assert len(calls) == 1


# --- result_view and trials_chart


def test_result_view(saved: SavedHeartRateResult, trials: pd.DataFrame) -> None:
    view = result_view(saved, trials)

    assert view.has_completed_trials
    assert (view.n_zones, view.best_n_zones) == (2, 3)  # 0.30 <= 0.28 + 0.03
    assert view.threshold == pytest.approx(0.31)
    assert view.cuts == [140]
    assert [(z.label, z.low, z.high) for z in view.zones] == [
        ("Z0", 0, 140),
        ("Z1", 140, 250),
    ]
    assert [row.chosen for row in view.summary] == [True, False]
    assert [row.n_pruned for row in view.summary] == [0, 1]
    assert view.zone_counts == [2, 3]
    assert view.reference is None
    assert view.best_zones_chart is not None
    assert view.best_zones_chart.error is None
    json.dumps(view.model_dump(mode="json"), allow_nan=False)


def test_result_view_with_reference(
    saved: SavedHeartRateResult, trials: pd.DataFrame
) -> None:
    view = result_view(saved, trials, reference=(0.35, 0.01))

    assert view.reference == (0.35, 0.01)
    assert view.summary[0].delta_vs_reference == pytest.approx(0.30 - 0.35)


def test_result_view_without_completed_trial(
    saved: SavedHeartRateResult, trials: pd.DataFrame
) -> None:
    view = result_view(saved, trials.assign(state="PRUNED", rmse=np.nan))

    assert not view.has_completed_trials
    assert view.summary == []


def test_result_view_missing_column_raises(
    saved: SavedHeartRateResult, trials: pd.DataFrame
) -> None:
    with pytest.raises(ValueError):
        result_view(saved, trials.drop(columns="state"))


def test_summary_rows_replace_nan() -> None:
    summary = pd.DataFrame(
        {
            "trial_number": [1],
            "cuts": [[150]],
            "rmse": [np.nan],
            "rmse_se": [np.nan],
            "n_complete": [0],
            "n_pruned": [3],
        },
        index=pd.Index([2], name="n_zones"),
    )

    (row,) = summary_rows(summary, chosen_n_zones=2)

    assert row.rmse is None
    assert row.chosen


def test_trials_chart(saved: SavedHeartRateResult, trials: pd.DataFrame) -> None:
    assert trials_chart(saved, trials, 3).error is None


def test_trials_chart_unknown_n_reports_the_error(
    saved: SavedHeartRateResult, trials: pd.DataFrame
) -> None:
    chart = trials_chart(saved, trials, 5)

    assert chart.figure is None
    assert chart.error
