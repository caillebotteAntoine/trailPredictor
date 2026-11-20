"""Heart-rate zone optimisations served by the API.

UI-independent: the routes of `TrailPredictor.api.heart_rate_routes` only
validate the requests and translate the errors into HTTP statuses.
"""

import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel

from TrailPredictor.api.dataset_service import DatasetSelection, DatasetService
from TrailPredictor.api.figures import ChartView, build_chart
from TrailPredictor.dataset import Dataset
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.modeling.heart_rate import (
    HR_CEIL_BPM,
    HR_FLOOR_BPM,
    MIN_ZONES,
    HeartRateConfig,
    ModelingHeartRate,
    SavedHeartRateResult,
    cuts_to_zones,
    list_saved_results,
    reference_score,
    run_optimization,
    select_n_zones_one_se,
    summarize_zone_counts,
)
from TrailPredictor.plot import PlotHeartRate
from TrailPredictor.plot.heart_rate import ZONE_COLORS
from TrailPredictor.speed_columns import Target
from TrailPredictor.utils import MODELS_DIR

logger = logging.getLogger(__name__)

MAX_ZONE_COUNT = len(ZONE_COLORS)  # one colour per zone in the figures
CHART_HEIGHT_PX = 700
MAX_REFERENCE_SCORES = 16


class UnknownResultError(LookupError):
    """Raised when a result key is not a saved optimisation."""


class NoCurrentDatasetError(LookupError):
    """Raised when an optimisation needs the dataset of the dataset page."""


class FormLimits(BaseModel):
    """Bounds of the optimisation form.

    Attributes
    ----------
    min_zones, max_zones : int
        Range of the numbers of zones.
    hr_floor_bpm, hr_ceil_bpm : int
        Extreme heart rates; the cuts lie strictly between them.
    """

    min_zones: int = MIN_ZONES
    max_zones: int = MAX_ZONE_COUNT
    hr_floor_bpm: int = HR_FLOOR_BPM
    hr_ceil_bpm: int = HR_CEIL_BPM


class CurrentDataset(BaseModel):
    """Dataset of the dataset page, as shown in the zones page.

    Attributes
    ----------
    selection : DatasetSelection
        Parameters, target and zones of the dataset.
    n_rows : int
        Number of segments.
    """

    selection: DatasetSelection
    n_rows: int


class PageContext(BaseModel):
    """What the zones page needs before any choice.

    Attributes
    ----------
    dataset : CurrentDataset or None
        Dataset of the dataset page, None if none was chosen.
    defaults : HeartRateConfig
        Default configuration of the optimisation form.
    limits : FormLimits
        Bounds of the optimisation form.
    """

    dataset: CurrentDataset | None
    defaults: HeartRateConfig
    limits: FormLimits


class SavedResultView(BaseModel):
    """Saved optimisation offered in the selector.

    Attributes
    ----------
    key : str
        Path of the trials, identifying the optimisation.
    target_name : Target
        Target of the optimisation.
    dataset_label : str
        Dataset the optimisation was run on.
    config : HeartRateConfig
        Configuration of the optimisation.
    config_key : str
        Hash of the configuration.
    """

    key: str
    target_name: Target
    dataset_label: str
    config: HeartRateConfig
    config_key: str


class OptimizationStarted(BaseModel):
    """Response of the optimisation route.

    Attributes
    ----------
    job_id : str
        Job running (or reloading) the optimisation.
    key : str
        Key of the result, once the job succeeded.
    already_saved : bool
        True if the result exists and is only reloaded.
    """

    job_id: str
    key: str
    already_saved: bool


class SummaryRow(BaseModel):
    """Best trial and trial counts of a number of zones."""

    n_zones: int
    trial_number: int
    cuts: list[int]
    rmse: float | None
    rmse_se: float | None
    n_complete: int
    n_pruned: int
    delta_vs_reference: float | None = None
    chosen: bool


class Zone(BaseModel):
    """Heart-rate zone of the chosen cuts, bounds in bpm."""

    label: str
    low: int
    high: int


class ResultView(BaseModel):
    """Metrics, table and main chart of an optimisation.

    Attributes
    ----------
    saved : SavedResultView
        The optimisation.
    has_completed_trials : bool
        False when every trial was pruned: nothing else is filled.
    n_zones : int or None
        Number of zones chosen by the one-standard-error rule.
    best_n_zones : int or None
        Number of zones of the lowest RMSE.
    threshold : float or None
        Lowest RMSE plus its standard error.
    cuts : list of int
        Cuts of the chosen zones, in bpm.
    zones : list of Zone
        Zones of `cuts`.
    reference : tuple of (float, float) or None
        RMSE and standard error of the reference zones on the dataset of the
        dataset page, None without it.
    summary : list of SummaryRow
        One row per number of zones.
    zone_counts : list of int
        Numbers of zones with completed trials.
    best_zones_chart : ChartView or None
        Best trial of each number of zones.
    """

    saved: SavedResultView
    has_completed_trials: bool
    n_zones: int | None = None
    best_n_zones: int | None = None
    threshold: float | None = None
    cuts: list[int] = []
    zones: list[Zone] = []
    reference: tuple[float, float] | None = None
    summary: list[SummaryRow] = []
    zone_counts: list[int] = []
    best_zones_chart: ChartView | None = None


def _none_if_nan(value: Any) -> Any:
    return None if isinstance(value, float) and math.isnan(value) else value


def saved_result_view(saved: SavedHeartRateResult) -> SavedResultView:
    """Describe a saved optimisation.

    Parameters
    ----------
    saved : SavedHeartRateResult
        Saved optimisation.

    Returns
    -------
    SavedResultView
        Its description, keyed by the path of its trials.
    """
    return SavedResultView(
        key=str(saved.result_path),
        target_name=saved.target_name,
        dataset_label=saved.dataset_label,
        config=saved.config,
        config_key=saved.config.key,
    )


def summary_rows(summary: pd.DataFrame, chosen_n_zones: int) -> list[SummaryRow]:
    """Convert the summary of `summarize_zone_counts` to table rows.

    Parameters
    ----------
    summary : pandas.DataFrame
        Indexed by ``n_zones``.
    chosen_n_zones : int
        Number of zones chosen by the one-standard-error rule.

    Returns
    -------
    list of SummaryRow
        One row per number of zones, NaN replaced by None.
    """
    records = summary.reset_index().to_dict("records")
    return [
        SummaryRow.model_validate(
            {str(key): _none_if_nan(value) for key, value in record.items()}
            | {"chosen": record["n_zones"] == chosen_n_zones}
        )
        for record in records
    ]


def result_view(
    saved: SavedHeartRateResult,
    trials: pd.DataFrame,
    reference: tuple[float, float] | None = None,
) -> ResultView:
    """Build the metrics, table and main chart of an optimisation.

    Parameters
    ----------
    saved : SavedHeartRateResult
        The optimisation.
    trials : pandas.DataFrame
        Its trials, as read by `SavedHeartRateResult.load_trials`.
    reference : tuple of (float, float) or None, optional
        RMSE and standard error of the reference zones. Default is None.

    Returns
    -------
    ResultView
        The view; only `saved` and `has_completed_trials` are filled when
        no trial was completed.

    Raises
    ------
    ValueError
        If the trials miss a column.
    """
    reference_rmse = None if reference is None else reference[0]
    plot = PlotHeartRate(trials, saved.target_name, reference_rmse=reference_rmse)
    view = ResultView(
        saved=saved_result_view(saved),
        has_completed_trials=not plot.completed_trials.dropna(subset=["rmse"]).empty,
        reference=reference,
    )
    if not view.has_completed_trials:
        return view
    summary = summarize_zone_counts(trials, plot.best_trials, reference_rmse)
    selection = select_n_zones_one_se(summary)
    # PlotHeartRate parsed every "cuts" cell into a list of int
    chosen = cast(list[int], summary.loc[selection.n_zones, "cuts"])
    cuts = [int(cut) for cut in chosen]
    figure = build_chart(
        "best zones per n",
        lambda: plot.best_zones_per_n().update_layout(height=CHART_HEIGHT_PX),
    )
    return view.model_copy(
        update={
            "n_zones": selection.n_zones,
            "best_n_zones": selection.best_n_zones,
            "threshold": selection.threshold,
            "cuts": cuts,
            "zones": [
                Zone(label=label, low=low, high=high)
                for label, (low, high) in cuts_to_zones(cuts).items()
            ],
            "summary": summary_rows(summary, selection.n_zones),
            "zone_counts": sorted(
                int(n) for n in plot.completed_trials["n_zones"].unique()
            ),
            "best_zones_chart": figure,
        }
    )


def trials_chart(
    saved: SavedHeartRateResult, trials: pd.DataFrame, n_zones: int
) -> ChartView:
    """Build the chart of every trial of a number of zones.

    Parameters
    ----------
    saved : SavedHeartRateResult
        The optimisation.
    trials : pandas.DataFrame
        Its trials.
    n_zones : int
        Number of zones to show.

    Returns
    -------
    ChartView
        The figure, or the error (e.g. no trial for `n_zones`).
    """
    plot = PlotHeartRate(trials, saved.target_name)
    return build_chart(
        f"trials of N={n_zones}",
        lambda: plot.trials_zones(n_zones).update_layout(height=CHART_HEIGHT_PX),
    )


class HeartRateService:
    """Saved optimisations, new optimisations and reference scores.

    Parameters
    ----------
    datasets : DatasetService
        Gives the dataset chosen in the dataset page.
    models_dir : pathlib.Path, optional
        Root directory of the saved results. Default is ``MODELS_DIR``.
    """

    def __init__(self, datasets: DatasetService, models_dir: Path = MODELS_DIR) -> None:
        self._datasets = datasets
        self._models_dir = models_dir
        self._references: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def current_dataset(self) -> tuple[DatasetSelection, Dataset] | None:
        """Return the dataset chosen in the dataset page.

        Returns
        -------
        tuple or None
            The `DatasetSelection` and its `Dataset` (with its target and
            zones), or None if no dataset was chosen or it was evicted from
            the cache.
        """
        selection = self._datasets.current
        if selection is None:
            return None
        try:
            dataset = self._datasets.dataset(
                selection.params, selection.target, selection.cuts
            )
        except LookupError:  # evicted since it was chosen
            logger.info("The current dataset %s is no longer loaded", selection)
            return None
        return selection, dataset

    def context(self) -> PageContext:
        """Return what the zones page needs before any choice.

        Returns
        -------
        PageContext
            Current dataset, form defaults and bounds.
        """
        current = self.current_dataset()
        dataset = (
            None
            if current is None
            else CurrentDataset(selection=current[0], n_rows=len(current[1].data))
        )
        return PageContext(
            dataset=dataset, defaults=HeartRateConfig(), limits=FormLimits()
        )

    def saved_results(self) -> list[SavedHeartRateResult]:
        """List the saved optimisations.

        Returns
        -------
        list of SavedHeartRateResult
            Results found in the models directory, read again at each call.
        """
        return list_saved_results(self._models_dir)

    def saved_result(self, key: str) -> SavedHeartRateResult:
        """Return the saved optimisation of a key.

        Only the listed results can be read, so a key cannot point the
        server to another file.

        Parameters
        ----------
        key : str
            Path of the trials, as in `SavedResultView.key`.

        Returns
        -------
        SavedHeartRateResult
            The optimisation.

        Raises
        ------
        UnknownResultError
            If `key` is not a saved optimisation.
        """
        for saved in self.saved_results():  # few files: loop over the results
            if str(saved.result_path) == key:
                return saved
        raise UnknownResultError(key)

    def start_optimization(
        self,
        target: Target,
        config: HeartRateConfig,
        submit: Callable[[Callable[[ProgressCallback], object]], str],
    ) -> OptimizationStarted:
        """Run (or reload) an optimisation on the current dataset in a job.

        Parameters
        ----------
        target : Target
            Target the RMSE is computed on.
        config : HeartRateConfig
            Search space and budget.
        submit : callable
            Submits a task and returns its job identifier, e.g.
            `JobManager.submit`.

        Returns
        -------
        OptimizationStarted
            Job identifier and key of the result.

        Raises
        ------
        NoCurrentDatasetError
            If no dataset was chosen in the dataset page.
        """
        current = self.current_dataset()
        if current is None:
            raise NoCurrentDatasetError("Choose a dataset in the dataset page first")
        _, dataset = current
        model = ModelingHeartRate(dataset.with_target(target), config, self._models_dir)
        job_id = submit(
            lambda report: run_optimization(
                dataset, target, config, progress_cb=report, models_dir=self._models_dir
            )
        )
        return OptimizationStarted(
            job_id=job_id,
            key=str(model.result_path),
            already_saved=model.result is not None,
        )

    def reference(self, saved: SavedHeartRateResult) -> tuple[float, float] | None:
        """Cross-validate the reference zones on the current dataset (cached).

        Parameters
        ----------
        saved : SavedHeartRateResult
            Optimisation giving the target and the folds.

        Returns
        -------
        tuple or None
            RMSE and standard error (floats), or None without current
            dataset.
        """
        current = self.current_dataset()
        if current is None:
            return None
        selection, dataset = current
        key = "|".join(
            [
                selection.model_dump_json(),
                saved.target_name,
                saved.config.model_dump_json(),
            ]
        )
        with self._lock:
            score = self._references.get(key)
        if score is None:
            score = reference_score(dataset, saved.target_name, saved.config)
            with self._lock:
                self._references[key] = score
                if len(self._references) > MAX_REFERENCE_SCORES:
                    self._references.popitem(last=False)
        return score
