"""Tests for :class:`TrailPredictor.plot.PlotHeartRate`."""

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.heart_rate import (
    HR_CEIL_BPM,
    REFERENCE_CUTS,
    ModelingHeartRate,
)
from TrailPredictor.plot import PlotHeartRate

TARGET_NAME = "pace"
REFERENCE_RMSE = 0.5


@pytest.fixture
def trials() -> pd.DataFrame:
    """Trials of a 2-zone and a 3-zone study; one pruned 3-zone trial."""
    return pd.DataFrame(
        {
            "n_zones": [2, 2, 2, 3, 3, 3],
            "trial_number": [0, 1, 2, 0, 1, 3],
            "rmse": [0.40, 0.30, 0.35, 0.28, np.nan, 0.32],
            "rmse_se": [0.01, 0.02, 0.01, 0.03, np.nan, 0.02],
            "cuts": [[150], [140], [145], [140, 170], [130, 200], [150, 175]],
            "state": ["COMPLETE"] * 4 + ["PRUNED", "COMPLETE"],
        }
    )


@pytest.fixture
def plot(trials: pd.DataFrame) -> PlotHeartRate:
    return PlotHeartRate(trials, TARGET_NAME, reference_rmse=REFERENCE_RMSE)


def _traces(fig, name: str) -> list:
    return [trace for trace in fig.data if trace.name == name]


def test_missing_column_raises(trials: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="state"):
        PlotHeartRate(trials.drop(columns="state"), TARGET_NAME)


def test_cuts_parsed_from_csv_strings(trials: pd.DataFrame) -> None:
    from_csv = trials.assign(cuts=trials["cuts"].map(str))

    best = PlotHeartRate(from_csv, TARGET_NAME).best_trials

    assert best.loc[3, "cuts"] == [140, 170]


def test_best_trials(plot: PlotHeartRate) -> None:
    best = plot.best_trials

    assert best.index.tolist() == [2, 3]
    assert best["trial_number"].tolist() == [1, 0]
    np.testing.assert_allclose(best["rmse"], [0.30, 0.28])


def test_trials_zones_bars_and_running_min(plot: PlotHeartRate) -> None:
    fig = plot.trials_zones(3)

    # pruned trial 1 is skipped: zones of trials 0 and 3 only
    z1 = _traces(fig, "Z1")[0]
    np.testing.assert_allclose(np.asarray(z1.base, dtype=float), [140, 150])
    np.testing.assert_allclose(z1.y, [30, 25])
    z2 = _traces(fig, "Z2")[0]
    np.testing.assert_allclose(z2.y, [HR_CEIL_BPM - 170, HR_CEIL_BPM - 175])
    np.testing.assert_allclose(_traces(fig, "RMSE min (N=3)")[0].y, [0.28, 0.28])
    assert list(fig.layout.xaxis2.ticktext) == ["0", "3"]


def test_trials_zones_unknown_n_raises(plot: PlotHeartRate) -> None:
    with pytest.raises(ValueError, match="7 zones"):
        plot.trials_zones(7)


def test_best_zones_per_n_pads_missing_zones(plot: PlotHeartRate) -> None:
    fig = plot.best_zones_per_n()

    # Z2 only exists for N=3
    z2 = _traces(fig, "Z2")[0]
    assert np.isnan(z2.y[0])
    assert z2.y[1] == pytest.approx(HR_CEIL_BPM - 170)
    rmse = _traces(fig, "RMSE meilleur essai ± SE")[0]
    np.testing.assert_allclose(rmse.error_y.array, [0.02, 0.03])


def test_reference_lines(plot: PlotHeartRate) -> None:
    fig = plot.best_zones_per_n()

    reference_cuts = _traces(fig, "Seuils référence")
    assert [trace.y[0] for trace in reference_cuts] == REFERENCE_CUTS
    assert sum(trace.showlegend for trace in reference_cuts) == 1
    reference_rmse = _traces(fig, f"RMSE référence (N={len(REFERENCE_CUTS) + 1})")
    assert reference_rmse[0].y[0] == pytest.approx(REFERENCE_RMSE)


def test_no_reference_rmse_line_without_value(trials: pd.DataFrame) -> None:
    fig = PlotHeartRate(trials, TARGET_NAME).trials_zones(2)

    assert not [trace for trace in fig.data if "référence (N=" in (trace.name or "")]


@pytest.fixture
def heart_rate_model(dataset: pd.DataFrame, tmp_path) -> ModelingHeartRate:
    """Model without saved result, isolated from the real models directory."""
    return ModelingHeartRate(Dataset(dataset, target=TARGET_NAME), models_dir=tmp_path)


def test_model_plot_property(
    heart_rate_model: ModelingHeartRate, trials: pd.DataFrame
) -> None:
    heart_rate_model.result = trials
    heart_rate_model.reference_rmse = REFERENCE_RMSE

    fig = heart_rate_model.plot.best_zones_per_n()

    reference_rmse = _traces(fig, f"RMSE référence (N={len(REFERENCE_CUTS) + 1})")
    assert reference_rmse[0].y[0] == pytest.approx(REFERENCE_RMSE)


def test_model_plot_without_result_raises(heart_rate_model: ModelingHeartRate) -> None:
    with pytest.raises(ValueError, match="no result"):
        _ = heart_rate_model.plot
