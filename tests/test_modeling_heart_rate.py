"""Tests for :mod:`TrailPredictor.modeling.heart_rate`."""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import FixedTrial
from pydantic import ValidationError

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.heart_rate import (
    HR_CEIL_BPM,
    HR_FLOOR_BPM,
    REFERENCE_CUTS,
    HeartRateConfig,
    ModelingHeartRate,
    OneSeSelection,
    cuts_to_zones,
    list_saved_results,
    select_n_zones_one_se,
    suggest_cuts,
    summarize_zone_counts,
)
from TrailPredictor.modeling.models import Modeling

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
def model(
    dataset: pd.DataFrame, config: HeartRateConfig, tmp_path: Path
) -> ModelingHeartRate:
    return ModelingHeartRate(Dataset(dataset), config, models_dir=tmp_path)


# --- cuts_to_zones


def test_cuts_to_zones() -> None:
    assert cuts_to_zones([142, 161]) == {
        "Z0": (HR_FLOOR_BPM, 142),
        "Z1": (142, 161),
        "Z2": (161, HR_CEIL_BPM),
    }


def test_cuts_to_zones_reference_cuts_are_valid() -> None:
    assert len(cuts_to_zones(REFERENCE_CUTS)) == len(REFERENCE_CUTS) + 1


@pytest.mark.parametrize(
    "cuts", [[161, 142], [150, 150], [HR_FLOOR_BPM], [HR_CEIL_BPM], [-1]]
)
def test_cuts_to_zones_invalid_cuts_raise(cuts: list[int]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        cuts_to_zones(cuts)


# --- HeartRateConfig


def test_config_coerces_list_of_zone_counts() -> None:
    assert HeartRateConfig(zone_counts=[2, 3]).zone_counts == (2, 3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("zone_counts", ()),
        ("zone_counts", (1,)),
        ("first_cut_range", (170, 120)),
        ("gap_range", (0, 10)),
        ("max_last_cut", HR_CEIL_BPM),
        ("n_trials", 0),
        ("n_splits", 1),
    ],
)
def test_config_invalid_field_raises(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        HeartRateConfig(**{field: value})


def test_config_first_cut_above_max_last_cut_raises() -> None:
    with pytest.raises(ValidationError, match="max_last_cut"):
        HeartRateConfig(first_cut_range=(190, 200), max_last_cut=180)


def test_config_typo_raises() -> None:
    with pytest.raises(ValidationError, match="first_cur_range"):
        HeartRateConfig(first_cur_range=(120, 170))


# --- suggest_cuts


@pytest.mark.parametrize(
    ("n_zones", "params", "expected"),
    [
        (2, {"c1": 150}, [150]),
        (4, {"c1": 140, "gap_2": 10, "gap_3": 5}, [140, 150, 155]),
    ],
)
def test_suggest_cuts(
    n_zones: int, params: dict[str, int], expected: list[int]
) -> None:
    cuts = suggest_cuts(FixedTrial(params), n_zones, HeartRateConfig())

    assert cuts == expected


# --- ModelingHeartRate


def test_modeling_factory_passes_config(
    dataset: pd.DataFrame, config: HeartRateConfig, tmp_path: Path
) -> None:
    model = Modeling(Dataset(dataset), models_dir=tmp_path).heart_rate(config)

    assert model.config is config
    assert model.result is None


def test_evaluate_cuts_zones(model: ModelingHeartRate) -> None:
    rmse, rmse_se = model.evaluate_cuts_zones(REFERENCE_CUTS)

    assert np.isfinite(rmse) and rmse > 0.0
    assert np.isfinite(rmse_se) and rmse_se >= 0.0


def test_evaluate_cuts_zones_does_not_modify_dataset(model: ModelingHeartRate) -> None:
    before = model.dataset.data

    model.evaluate_cuts_zones([100, 200])

    pd.testing.assert_frame_equal(model.dataset.data, before)


def test_optimize_zones_result(
    model: ModelingHeartRate, config: HeartRateConfig
) -> None:
    studies = model.optimize_zones()

    assert set(studies) == set(config.zone_counts)
    assert model.result is not None
    assert model.result.index.is_unique
    completed = model.result.loc[model.result["state"] == "COMPLETE"]
    assert completed.groupby("n_zones").size().tolist() == [config.n_trials] * 2
    assert completed["cuts"].map(len).add(1).equals(completed["n_zones"])


def test_optimize_zones_reports_progress(
    model: ModelingHeartRate, config: HeartRateConfig
) -> None:
    reports: list[tuple[float, str]] = []

    model.optimize_zones(
        progress_cb=lambda fraction, text: reports.append((fraction, text))
    )

    fractions = [fraction for fraction, _ in reports]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert all(0.0 <= fraction <= 1.0 for fraction in fractions)
    assert (
        reports[-1][1] == f"N=3: {config.n_trials}/{config.n_trials} completed trials"
    )


def test_optimize_zones_is_reproducible(
    dataset: pd.DataFrame, config: HeartRateConfig, tmp_path: Path
) -> None:
    first = ModelingHeartRate(Dataset(dataset), config, models_dir=tmp_path / "a")
    second = ModelingHeartRate(Dataset(dataset), config, models_dir=tmp_path / "b")

    first.optimize_zones()
    second.optimize_zones()

    assert first.result is not None and second.result is not None
    pd.testing.assert_frame_equal(first.result, second.result)


def test_optimize_zones_saves_and_reloads(
    model: ModelingHeartRate, tmp_path: Path
) -> None:
    model.optimize_zones()

    reloaded = ModelingHeartRate(model.dataset, model.config, models_dir=tmp_path)

    assert model.result_path.exists()
    assert model.result_path.with_suffix(".json").exists()
    assert reloaded.result is not None
    assert len(reloaded.result) == len(model.result)


def test_cuts_above_max_last_cut_are_pruned(
    dataset: pd.DataFrame, tmp_path: Path
) -> None:
    config = HeartRateConfig(
        zone_counts=(2,),
        first_cut_range=(150, 200),
        max_last_cut=160,
        n_trials=2,
        max_trials_factor=10,
        n_splits=N_SPLITS,
    )
    model = ModelingHeartRate(Dataset(dataset), config, models_dir=tmp_path)

    model.optimize_zones()

    assert model.result is not None
    pruned = model.result.loc[model.result["state"] == "PRUNED"]
    assert pruned["cuts"].map(lambda cuts: cuts[-1] > 160).all()
    assert pruned["rmse"].isna().all()


# --- list_saved_results


def test_list_saved_results_empty_dir(tmp_path: Path) -> None:
    assert not list_saved_results(tmp_path)


def test_list_saved_results_finds_saved_model(
    model: ModelingHeartRate, tmp_path: Path
) -> None:
    model.result = pd.DataFrame({"n_zones": [2]})
    model.save_result()

    saved = list_saved_results(tmp_path)

    assert len(saved) == 1
    assert saved[0].target_name == model.dataset.target_name
    assert saved[0].config == model.config
    assert saved[0].result_path == model.result_path


def test_list_saved_results_skips_invalid_files(
    model: ModelingHeartRate, tmp_path: Path
) -> None:
    model.result = pd.DataFrame({"n_zones": [2]})
    model.save_result()
    target_dir = model.result_path.parent
    (target_dir / "no_csv.json").write_text(model.config.model_dump_json())
    (target_dir / "bad.json").write_text('{"unknown_field": 1}')
    (target_dir / "bad.csv").write_text("n_zones\n2\n")
    unknown_target = tmp_path / ModelingHeartRate.name / "not_a_target"
    unknown_target.mkdir()
    (unknown_target / "key.json").write_text(model.config.model_dump_json())
    (unknown_target / "key.csv").write_text("n_zones\n2\n")

    saved = list_saved_results(tmp_path)

    assert [s.result_path for s in saved] == [model.result_path]


# --- summarize_zone_counts


@pytest.fixture
def trials() -> pd.DataFrame:
    """Trials of 2 and 3 zones, with a pruned trial for N=3."""
    return pd.DataFrame(
        {
            "n_zones": [2, 2, 3, 3, 3],
            "state": ["COMPLETE", "COMPLETE", "COMPLETE", "PRUNED", "COMPLETE"],
        }
    )


@pytest.fixture
def best_trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_number": [1, 4],
            "cuts": [[150], [140, 160]],
            "rmse": [0.55, 0.5],
            "rmse_se": [0.1, 0.25],
            "extra": ["dropped", "dropped"],
        },
        index=pd.Index([2, 3], name="n_zones"),
    )


def test_summarize_zone_counts(trials: pd.DataFrame, best_trials: pd.DataFrame) -> None:
    summary = summarize_zone_counts(trials, best_trials)

    expected = best_trials.drop(columns="extra").assign(
        n_complete=[2, 2], n_pruned=[0, 1]
    )
    pd.testing.assert_frame_equal(summary, expected)


def test_summarize_zone_counts_with_reference(
    trials: pd.DataFrame, best_trials: pd.DataFrame
) -> None:
    summary = summarize_zone_counts(trials, best_trials, reference_rmse=0.5)

    np.testing.assert_allclose(summary["delta_vs_reference"], [0.05, 0.0])


def test_summarize_zone_counts_empty() -> None:
    trials = pd.DataFrame({"n_zones": pd.Series(dtype=int), "state": []})
    best_trials = pd.DataFrame(
        columns=["trial_number", "cuts", "rmse", "rmse_se"],
        index=pd.Index([], name="n_zones"),
    )

    summary = summarize_zone_counts(trials, best_trials)

    assert summary.empty
    assert {"n_complete", "n_pruned"} <= set(summary.columns)


# --- select_n_zones_one_se


def _summary(rmse: list[float], rmse_se: list[float]) -> pd.DataFrame:
    index = pd.Index(range(2, 2 + len(rmse)), name="n_zones")
    return pd.DataFrame({"rmse": rmse, "rmse_se": rmse_se}, index=index)


@pytest.mark.parametrize(
    ("rmse", "rmse_se", "expected"),
    [
        # N=2 within one SE of the best N=3: the simpler model wins
        ([0.55, 0.5, 0.6], [0.1, 0.25, 0.1], OneSeSelection(2, 3, 0.75)),
        # N=2 above the threshold: the best N is kept
        ([1.0, 0.5, 0.6], [0.1, 0.25, 0.1], OneSeSelection(3, 3, 0.75)),
        # missing SE counts as 0
        ([0.55, 0.5], [0.1, np.nan], OneSeSelection(3, 3, 0.5)),
        # missing RMSE is ignored
        ([np.nan, 0.5], [0.1, 0.25], OneSeSelection(3, 3, 0.75)),
    ],
)
def test_select_n_zones_one_se(
    rmse: list[float], rmse_se: list[float], expected: OneSeSelection
) -> None:
    selection = select_n_zones_one_se(_summary(rmse, rmse_se))

    assert selection.n_zones == expected.n_zones
    assert selection.best_n_zones == expected.best_n_zones
    assert selection.threshold == pytest.approx(expected.threshold)


@pytest.mark.parametrize("rmse", [[], [np.nan]])
def test_select_n_zones_one_se_without_rmse_raises(rmse: list[float]) -> None:
    with pytest.raises(ValueError, match="finite RMSE"):
        select_n_zones_one_se(_summary(rmse, [np.nan] * len(rmse)))
