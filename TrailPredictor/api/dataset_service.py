"""Datasets served by the API: loading, feature table, charts and current dataset.

UI-independent: the routes of `TrailPredictor.api.dataset_routes` only
validate the requests and translate the errors into HTTP statuses.
"""

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import NamedTuple

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from TrailPredictor.api.figures import ChartView, build_chart
from TrailPredictor.dataset import Dataset
from TrailPredictor.feature_selection import (
    PERCENT,
    FeatureGroup,
    Preset,
    apply_filters,
    apply_preset,
    correlation_table,
)
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.modeling.heart_rate import (
    SavedHeartRateResult,
    cuts_to_zones,
    list_saved_results,
    saved_one_se_cuts,
)
from TrailPredictor.speed_columns import Target

logger = logging.getLogger(__name__)

MAX_LOADED_DATASETS = 2  # a dataset holds every segment of tens of activities
MAX_CORRELATION_TABLES = 16
CHART_HEIGHT_PX = 700  # 3 x 3 grid of the targets
TARGET_HISTOGRAM_TITLE = "Distribution de la cible"

DatasetLoader = Callable[[int, int, int, ProgressCallback], Dataset]


class DatasetNotLoadedError(LookupError):
    """Raised when a dataset is used before its loading."""


class DatasetParams(NamedTuple):
    """Parameters identifying a loaded dataset.

    Attributes
    ----------
    min_distance : int
        Minimum activity distance, in km (exclusive).
    max_distance : int
        Maximum activity distance, in km (exclusive).
    segment_width : int
        Segment length, in metres.
    """

    min_distance: int
    max_distance: int
    segment_width: int


class FeatureFilters(BaseModel):
    """Filters of the feature table (see `apply_filters`)."""

    search: str = ""
    groups: list[FeatureGroup] = []
    min_abs: float = 0.0
    max_missing: float = PERCENT


class FeatureRow(BaseModel):
    """Row of the feature table (see `correlation_table`)."""

    name: str
    spearman: float | None
    missing_pct: float
    group: FeatureGroup


class FeaturesView(BaseModel):
    """Feature table of a dataset.

    Attributes
    ----------
    n_rows : int
        Number of segments of the dataset.
    ranked : list of str
        Every numeric feature, before the filters, by decreasing absolute
        correlation.
    groups : list of FeatureGroup
        Groups present among the numeric features.
    rows : list of FeatureRow
        Features passing the filters, by decreasing absolute correlation.
    non_numeric : list of str
        Candidate features left out, as a scatter plot needs numbers.
    """

    n_rows: int
    ranked: list[str]
    groups: list[FeatureGroup]
    rows: list[FeatureRow]
    non_numeric: list[str]


class DatasetSelection(BaseModel):
    """Dataset chosen in the dataset page, used by the other pages.

    Attributes
    ----------
    min_distance, max_distance : int
        Activity distance range, in km.
    segment_width : int
        Segment length, in metres.
    target : Target
        Target of the models.
    cuts : list of int
        Heart-rate zone cuts, in bpm.
    """

    min_distance: int
    max_distance: int
    segment_width: int
    target: Target
    cuts: list[int]

    @property
    def params(self) -> DatasetParams:
        """Return the parameters identifying the loaded dataset.

        Returns
        -------
        DatasetParams
            Distance range and segment width.
        """
        return DatasetParams(self.min_distance, self.max_distance, self.segment_width)


class SavedZones(BaseModel):
    """Zones chosen by the one-standard-error rule of a saved optimisation.

    Attributes
    ----------
    key : str
        Path of the saved trials, identifying the optimisation.
    target_name : Target
        Target of the optimisation.
    dataset_label : str
        Dataset the optimisation was run on.
    config_key : str
        Hash of the optimisation configuration.
    cuts : list of int
        Cuts of the chosen zones, in bpm.
    """

    key: str
    target_name: Target
    dataset_label: str
    config_key: str
    cuts: list[int]


def features_view(
    dataset: Dataset, correlations: pd.DataFrame, filters: FeatureFilters
) -> FeaturesView:
    """Build the feature table shown in the dataset page.

    Parameters
    ----------
    dataset : Dataset
        Dataset whose candidate features are listed.
    correlations : pandas.DataFrame
        Table returned by `correlation_table` for this dataset.
    filters : FeatureFilters
        Filters of the table.

    Returns
    -------
    FeaturesView
        Filtered table and its context.
    """
    filtered = apply_filters(
        correlations,
        filters.search,
        [group.value for group in filters.groups],
        filters.min_abs,
        filters.max_missing,
    )
    rows = (
        filtered.astype({"spearman": object})
        .where(filtered.notna(), None)
        .to_dict("records")
    )
    ranked = correlations["name"].tolist()
    return FeaturesView(
        n_rows=len(dataset.data),
        ranked=ranked,
        groups=sorted(FeatureGroup(group) for group in correlations["group"].unique()),
        rows=[FeatureRow.model_validate(row) for row in rows],
        non_numeric=[name for name in dataset.features if name not in set(ranked)],
    )


def preset_features(
    correlations: pd.DataFrame, preset: Preset, filters: FeatureFilters
) -> list[str]:
    """Return the features of a preset, among the filtered features.

    Parameters
    ----------
    correlations : pandas.DataFrame
        Table returned by `correlation_table`.
    preset : Preset
        Predefined selection.
    filters : FeatureFilters
        Filters applied before the preset.

    Returns
    -------
    list of str
        Names of the features of the preset.
    """
    pool = apply_filters(
        correlations,
        filters.search,
        [group.value for group in filters.groups],
        filters.min_abs,
        filters.max_missing,
    )
    return apply_preset(preset, pool)


def dataset_chart(dataset: Dataset, feature: str | None = None) -> ChartView:
    """Build a chart of the dataset page, drawn for every target.

    Parameters
    ----------
    dataset : Dataset
        Dataset to plot. It is not modified.
    feature : str or None, optional
        Feature on the x axis of a target-vs-feature scatter plot. Default is
        None: the histogram of the target.

    Returns
    -------
    ChartView
        The 3 x 3 figure of the targets, or the error that prevented it.
    """
    plot = dataset.plot  # new object: merge_figure switches its target

    def build() -> go.Figure:
        """Draw the chart.

        Returns
        -------
        plotly.graph_objects.Figure
            The 3 x 3 figure of the targets.
        """
        if feature is None:
            figure = plot.merge_figure(
                plot.target_histogram, title=TARGET_HISTOGRAM_TITLE
            )
        else:
            figure = plot.merge_figure(
                plot.target_vs_feat, title=f"Cible vs {feature}", features=feature
            )
        return figure.update_layout(height=CHART_HEIGHT_PX)

    return build_chart(feature or "target histogram", build)


def saved_zones(saved_results: Sequence[SavedHeartRateResult]) -> list[SavedZones]:
    """Describe the zones of the saved heart-rate optimisations.

    Parameters
    ----------
    saved_results : sequence of SavedHeartRateResult
        Saved optimisations, e.g. from `list_saved_results`.

    Returns
    -------
    list of SavedZones
        One entry per optimisation whose trials could be read, in order.
    """
    cuts = saved_one_se_cuts(saved_results)
    return [
        SavedZones(
            key=str(saved.result_path),
            target_name=saved.target_name,
            dataset_label=saved.dataset_label,
            config_key=saved.config.key,
            cuts=cuts[saved.result_path],
        )
        for saved in saved_results
        if saved.result_path in cuts
    ]


def _load_dataset(
    min_distance: int,
    max_distance: int,
    segment_width: int,
    progress_cb: ProgressCallback,
) -> Dataset:
    return Dataset.load(
        min_distance=min_distance,
        max_distance=max_distance,
        segment_width=segment_width,
        progress_cb=progress_cb,
    )


class DatasetService:
    """Cache of the loaded datasets and dataset chosen in the dataset page.

    Parameters
    ----------
    loader : callable, optional
        Loads ``(min_distance, max_distance, segment_width, progress_cb)``
        into a `Dataset`. Default is `Dataset.load`.
    max_loaded : int, optional
        Number of datasets kept in memory, least recently used first evicted.
        Default is ``MAX_LOADED_DATASETS``.
    saved_results : callable, optional
        Lists the saved heart-rate optimisations. Default is
        `list_saved_results`.

    Notes
    -----
    The methods are thread-safe: the loading runs in the job threads while
    the routes read the cache. The cached datasets are never modified: the
    views work on copies (`Dataset.with_target`).
    """

    def __init__(
        self,
        loader: DatasetLoader = _load_dataset,
        max_loaded: int = MAX_LOADED_DATASETS,
        saved_results: Callable[[], list[SavedHeartRateResult]] = list_saved_results,
    ) -> None:
        self._loader = loader
        self._max_loaded = max_loaded
        self._saved_results = saved_results
        self._loaded: OrderedDict[DatasetParams, Dataset] = OrderedDict()
        self._correlations: OrderedDict[tuple[DatasetParams, Target], pd.DataFrame]
        self._correlations = OrderedDict()
        self._current: DatasetSelection | None = None
        self._lock = threading.Lock()

    def load(self, params: DatasetParams, progress_cb: ProgressCallback) -> None:
        """Load and cache a dataset, unless it is already cached.

        Parameters
        ----------
        params : DatasetParams
            Distance range and segment width.
        progress_cb : ProgressCallback
            Progress reporter of the loading.

        Raises
        ------
        ValueError
            If the distance range is empty.
        """
        with self._lock:
            if params in self._loaded:
                self._loaded.move_to_end(params)
                return
        # loaded outside the lock: the other datasets stay readable
        dataset = self._loader(*params, progress_cb)
        with self._lock:
            self._loaded[params] = dataset
            self._loaded.move_to_end(params)
            if len(self._loaded) > self._max_loaded:
                evicted, _ = self._loaded.popitem(last=False)
                logger.info("Evicted dataset %s from the cache", evicted)

    def dataset(
        self, params: DatasetParams, target: Target, cuts: Sequence[int] | None = None
    ) -> Dataset:
        """Return a loaded dataset with a target and optional zones.

        Parameters
        ----------
        params : DatasetParams
            Distance range and segment width.
        target : Target
            Target of the returned copy.
        cuts : sequence of int or None, optional
            Heart-rate zone cuts, in bpm. Default is None: the zones the
            dataset was built with.

        Returns
        -------
        Dataset
            Copy of the cached dataset.

        Raises
        ------
        DatasetNotLoadedError
            If the dataset was not loaded.
        ValueError
            If the cuts are invalid.
        """
        with self._lock:
            dataset = self._loaded.get(params)
            if dataset is not None:
                self._loaded.move_to_end(params)
        if dataset is None:
            raise DatasetNotLoadedError(f"Dataset {tuple(params)} is not loaded")
        dataset = dataset.with_target(target)
        if cuts is not None:
            dataset = dataset.with_heart_rate_zones(cuts_to_zones(cuts))
        return dataset

    def correlations(self, params: DatasetParams, target: Target) -> pd.DataFrame:
        """Return the feature table of a loaded dataset (cached).

        Parameters
        ----------
        params : DatasetParams
            Distance range and segment width.
        target : Target
            Target the features are correlated with.

        Returns
        -------
        pandas.DataFrame
            Table of `correlation_table`; do not modify it.

        Raises
        ------
        DatasetNotLoadedError
            If the dataset was not loaded.
        """
        key = (params, target)
        with self._lock:
            table = self._correlations.get(key)
        if table is None:
            table = correlation_table(self.dataset(params, target))
            with self._lock:
                self._correlations[key] = table
                if len(self._correlations) > MAX_CORRELATION_TABLES:
                    self._correlations.popitem(last=False)
        return table

    def saved_zones(self) -> list[SavedZones]:
        """Describe the zones of the saved heart-rate optimisations.

        Returns
        -------
        list of SavedZones
            Zones offered in the dataset page, read again at each call.
        """
        return saved_zones(self._saved_results())

    @property
    def current(self) -> DatasetSelection | None:
        """Return the dataset chosen in the dataset page.

        Returns
        -------
        DatasetSelection or None
            Last selection, or None if no dataset was chosen yet.
        """
        with self._lock:
            return self._current

    def select(self, selection: DatasetSelection) -> None:
        """Choose the dataset used by the other pages.

        Parameters
        ----------
        selection : DatasetSelection
            Loaded dataset, target and zones.

        Raises
        ------
        DatasetNotLoadedError
            If the dataset was not loaded.
        ValueError
            If the cuts are invalid.
        """
        self.dataset(selection.params, selection.target, selection.cuts)
        with self._lock:
            self._current = selection
