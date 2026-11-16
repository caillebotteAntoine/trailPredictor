"""Optimisation of the heart-rate zone cuts with Optuna."""

import logging
from collections.abc import Sequence
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Self

import numpy as np
import optuna
import pandas as pd
from optuna.study import MaxTrialsCallback
from optuna.trial import BaseTrial, TrialState
from pydantic import AfterValidator, Field, PositiveInt, model_validator
from sklearn.linear_model import LinearRegression

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.base_model import BaseModeling
from TrailPredictor.modeling.config import ModelConfig
from TrailPredictor.utils import MODELS_DIR

if TYPE_CHECKING:
    from TrailPredictor.plot.heart_rate import PlotHeartRate

logger = logging.getLogger(__name__)

HR_FLOOR_BPM, HR_CEIL_BPM = 0, 250  # extreme values of the heart rate (bpm)
MIN_ZONES = 2  # at least one cut
REFERENCE_CUTS = [142, 161, 170, 183]


def cuts_to_zones(cuts: Sequence[int]) -> dict[str, tuple[int, int]]:
    """Convert N-1 increasing cuts into N contiguous zones Z0 ... Z{N-1}.

    Parameters
    ----------
    cuts : sequence of int
        Strictly increasing cuts, in bpm, strictly between ``HR_FLOOR_BPM``
        and ``HR_CEIL_BPM``.

    Returns
    -------
    dict[str, tuple[int, int]]
        Zone label mapped to its ``(low, high)`` bounds, in bpm; the first
        zone starts at ``HR_FLOOR_BPM`` and the last ends at ``HR_CEIL_BPM``.

    Raises
    ------
    ValueError
        If the cuts are not strictly increasing within the bounds.

    Examples
    --------
    >>> cuts_to_zones([142, 161])
    {'Z0': (0, 142), 'Z1': (142, 161), 'Z2': (161, 250)}
    """
    bounds = [HR_FLOOR_BPM, *cuts, HR_CEIL_BPM]
    if not all(low < high for low, high in pairwise(bounds)):
        raise ValueError(
            f"Cuts must be strictly increasing within "
            f"({HR_FLOOR_BPM}, {HR_CEIL_BPM}) bpm: {list(cuts)}"
        )
    return {f"Z{k}": (low, high) for k, (low, high) in enumerate(pairwise(bounds))}


def _check_ordered(bounds: tuple[int, int]) -> tuple[int, int]:
    """Reject a ``(low, high)`` range with ``low > high``."""
    low, high = bounds
    if low > high:
        raise ValueError(f"Range lower bound {low} is above its upper bound {high}")
    return bounds


# (low, high) range of positive integers, bounds included
PositiveRange = Annotated[
    tuple[PositiveInt, PositiveInt], AfterValidator(_check_ordered)
]


class HeartRateConfig(ModelConfig):
    """Search space and budget of the heart-rate zone optimisation.

    Parameters
    ----------
    seed : int, optional
        Seed of the Optuna sampler. Default is 42.
    n_splits : int, optional
        Number of ``GroupKFold`` folds, at least 2. Default is 5.
    zone_counts : tuple of int, optional
        Numbers of zones to optimise, each at least ``MIN_ZONES``; one study
        per value. A list is accepted. Default is ``(5,)``.
    first_cut_range : tuple of (int, int), optional
        Range of the first cut, in bpm. Default is ``(120, 170)``.
    gap_range : tuple of (int, int), optional
        Range of the gap between two successive cuts, in bpm. Default is
        ``(4, 30)``.
    max_last_cut : int, optional
        Trials whose last cut is above this value (outliers) are pruned, in
        bpm. Default is 188.
    n_trials : int, optional
        Number of completed (not pruned) trials per study. Default is 10.
    max_trials_factor : int, optional
        Bound of the total number of trials (pruned included), as a multiple
        of `n_trials`. Default is 20.

    Raises
    ------
    pydantic.ValidationError
        If a field is unknown or invalid, or if no first cut can be below
        `max_last_cut`.
    """

    zone_counts: tuple[Annotated[int, Field(ge=MIN_ZONES)], ...] = Field(
        default=(5,), min_length=1
    )
    first_cut_range: PositiveRange = (120, 170)
    gap_range: PositiveRange = (4, 30)
    max_last_cut: int = Field(default=188, gt=HR_FLOOR_BPM, lt=HR_CEIL_BPM)
    n_trials: PositiveInt = 10
    max_trials_factor: PositiveInt = 20

    @model_validator(mode="after")
    def _check_first_cut_below_max(self) -> Self:
        """Reject a search space where every trial would be pruned."""
        if self.first_cut_range[0] > self.max_last_cut:
            raise ValueError(
                f"first_cut_range {self.first_cut_range} is above "
                f"max_last_cut {self.max_last_cut}"
            )
        return self


def suggest_cuts(trial: BaseTrial, n_zones: int, config: HeartRateConfig) -> list[int]:
    """Suggest N-1 increasing cuts: a first cut, then positive gaps.

    Parameters
    ----------
    trial : optuna.trial.BaseTrial
        Trial suggesting the parameters ``c1``, ``gap_2`` ... ``gap_{N-1}``.
    n_zones : int
        Number of zones N.
    config : HeartRateConfig
        Ranges of the first cut and of the gaps.

    Returns
    -------
    list of int
        Strictly increasing cuts, in bpm.
    """
    cuts = [trial.suggest_int("c1", *config.first_cut_range)]
    for k in range(2, n_zones):  # loop over the search parameters, not the data
        cuts.append(cuts[-1] + trial.suggest_int(f"gap_{k}", *config.gap_range))
    return cuts


class ModelingHeartRate(BaseModeling[HeartRateConfig]):
    """Optimisation of the heart-rate zone cuts.

    For each number of zones, an Optuna study searches the cuts minimising
    the cross-validated RMSE of a linear regression whose predictors include
    the heart-rate zone of each segment.

    Parameters
    ----------
    dataset : Dataset
        Dataset the zones are optimised on. It is not modified.
    config : HeartRateConfig or None, optional
        Search space and budget. Default is None, meaning
        ``HeartRateConfig()``.
    models_dir : pathlib.Path, optional
        Root directory of the saved results. Default is ``MODELS_DIR``.

    Attributes
    ----------
    result : pandas.DataFrame or None
        One row per trial (columns ``n_zones``, ``trial_number``, ``rmse``,
        ``rmse_se``, ``cuts``, ``state``), or None before any run.
    reference_rmse : float or None
        RMSE of the ``REFERENCE_CUTS`` zones, drawn by the plots when set.
    """

    name: ClassVar[str] = "heart_rate"

    def __init__(
        self,
        dataset: Dataset,
        config: HeartRateConfig | None = None,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        super().__init__(
            dataset, HeartRateConfig() if config is None else config, models_dir
        )
        self.reference_rmse: float | None = None

    @property
    def plot(self) -> "PlotHeartRate":
        """Return a plotting interface of the optimisation trials.

        Returns
        -------
        PlotHeartRate
            Plots of ``result``, with ``reference_rmse`` as reference line.

        Raises
        ------
        ValueError
            If the model has no result yet.
        """
        # Lazy import: TrailPredictor.plot.heart_rate imports this module.
        from TrailPredictor.plot.heart_rate import (  # pylint: disable=import-outside-toplevel
            PlotHeartRate,
        )

        return PlotHeartRate.from_model(self, reference_rmse=self.reference_rmse)

    def make_estimator(self) -> LinearRegression:
        """Return the unfitted linear regression.

        Returns
        -------
        sklearn.linear_model.LinearRegression
            Unfitted regressor.
        """
        return LinearRegression()

    def evaluate_cuts_zones(self, cuts: Sequence[int]) -> tuple[float, float]:
        """Cross-validate the model with the zones defined by `cuts`.

        The zones are rebuilt on a copy of the dataset, which is unchanged.

        Parameters
        ----------
        cuts : sequence of int
            Strictly increasing cuts, in bpm.

        Returns
        -------
        rmse : float
            Mean RMSE over the folds.
        rmse_se : float
            Standard error of the mean RMSE.

        Raises
        ------
        ValueError
            If the cuts are not strictly increasing within the bounds.
        """
        zoned = self._dataset.with_heart_rate_zones(cuts_to_zones(cuts))
        fold_rmse = -self.cross_validate(zoned)["test_score"]
        return float(fold_rmse.mean()), float(
            fold_rmse.std(ddof=1) / np.sqrt(fold_rmse.size)
        )

    def _objective(self, trial: optuna.Trial, n_zones: int) -> float:
        """Mean RMSE of the zones suggested by `trial`."""
        cuts = suggest_cuts(trial, n_zones, self._config)
        trial.set_user_attr("cuts", cuts)
        if cuts[-1] > self._config.max_last_cut:
            raise optuna.TrialPruned()  # pruned before any costly computation
        rmse, rmse_se = self.evaluate_cuts_zones(cuts)
        trial.set_user_attr("rmse_se", rmse_se)
        return rmse

    def optimize_n_zones(self, n_zones: int) -> optuna.Study:
        """Search the best cuts for a given number of zones.

        Parameters
        ----------
        n_zones : int
            Number of zones N, i.e. N-1 cuts.

        Returns
        -------
        optuna.Study
            Finished study, with the ``cuts`` and ``rmse_se`` of each trial in
            its user attributes.
        """
        study = optuna.create_study(
            study_name=f"heart_rate_zones_{n_zones}",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self._config.seed, multivariate=True
            ),
        )
        study.optimize(
            partial(self._objective, n_zones=n_zones),
            n_trials=self._config.n_trials * self._config.max_trials_factor,
            # stop after n_trials completed trials: the (free) pruned trials do
            # not consume the budget
            callbacks=[
                MaxTrialsCallback(self._config.n_trials, states=(TrialState.COMPLETE,))
            ],
        )
        return study

    def optimize_zones(self) -> dict[int, optuna.Study]:
        """Run one study per number of zones, then store and save the trials.

        Returns
        -------
        dict[int, optuna.Study]
            Finished study of each number of zones of ``config.zone_counts``.
        """
        studies = {
            n_zones: self.optimize_n_zones(n_zones)
            for n_zones in self._config.zone_counts
        }
        self.result = pd.concat(
            [
                pd.DataFrame(
                    {
                        "n_zones": n_zones,
                        "trial_number": [t.number for t in study.trials],
                        "rmse": [t.value for t in study.trials],
                        "rmse_se": [t.user_attrs.get("rmse_se") for t in study.trials],
                        "cuts": [t.user_attrs.get("cuts") for t in study.trials],
                        "state": [t.state.name for t in study.trials],
                    }
                )
                for n_zones, study in studies.items()
            ],
            ignore_index=True,
        )
        self.save_result()
        return studies
