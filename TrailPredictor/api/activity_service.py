"""Activities served by the API: catalogue, loaded activities and their views.

UI-independent: the routes of `TrailPredictor.api.activity_routes` only
validate the requests and translate the errors into HTTP statuses.
"""

import json
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from TrailPredictor.activities import RunningActivities
from TrailPredictor.activity_metrics import ActivityMetrics, compute_activity_metrics
from TrailPredictor.api.figures import ChartView, build_chart
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.loading import (
    activity_file_name,
    load_activities,
    parse_strava_date,
    running_activities,
)
from TrailPredictor.plot import PlotActivities
from TrailPredictor.plot.track_3d import TRACK_TOOLTIP, ColorBy, MapTexture

logger = logging.getLogger(__name__)

MAX_LOADED_ACTIVITIES = 8  # segmented activities kept in memory
METRES_PER_KM = 1000.0
ISO_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

XAxis = Literal["distance", "time"]
SummaryLoader = Callable[[], pd.DataFrame]
ActivityLoader = Callable[[str, int, ProgressCallback], RunningActivities]
# Charts of the activity page: name -> figure builder
CHARTS: dict[str, Callable[[PlotActivities, XAxis], Any]] = {
    "altitude": lambda plot, _x_axis: plot.altitude(),
    "heart_rate": lambda plot, x_axis: plot.hr_overlay(x_axis=x_axis),
    "pace": lambda plot, _x_axis: plot.pace_by_segment(),
    "heart_rate_zones": lambda plot, _x_axis: plot.time_in_hr_zones(),
}


class ActivityNotLoadedError(LookupError):
    """Raised when the views of an activity are requested before its loading."""


class ActivitySummary(BaseModel):
    """Row of the activity selector.

    Attributes
    ----------
    id : int
        Strava identifier.
    name : str
        Name of the activity.
    date : str or None
        Start date, ISO 8601, or None if it cannot be parsed.
    distance_km : float or None
        Distance, in km, or None if unknown.
    """

    id: int
    name: str
    date: str | None
    distance_km: float | None


class ChartsView(BaseModel):
    """Summary metrics and charts of an activity.

    Attributes
    ----------
    metrics : dict[str, float or None]
        Fields of `ActivityMetrics`; None when unavailable.
    charts : dict[str, ChartView]
        Chart of each name of ``CHARTS``.
    """

    metrics: dict[str, float | None]
    charts: dict[str, ChartView]


class TrackLegendView(BaseModel):
    """Colour legend of the 3D track (see `TrackLegend`)."""

    label: str
    vmin_text: str
    vmax_text: str
    css_gradient: str
    attribution: str


class TrackView(BaseModel):
    """3D map of the GPS track.

    Attributes
    ----------
    deck : dict
        deck.gl JSON of the map, rendered by ``createDeck``.
    tooltip : dict
        Tooltip settings of ``createDeck``.
    legend : TrackLegendView
        Colour legend of the track.
    """

    deck: dict[str, Any]
    tooltip: dict[str, str] = TRACK_TOOLTIP
    legend: TrackLegendView


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _none_if_missing(values: pd.Series) -> pd.Series:
    """Return `values` as objects, missing values replaced by None (JSON null)."""
    return values.astype(object).where(values.notna(), None)


def summarize_activities(activities: pd.DataFrame) -> list[ActivitySummary]:
    """Describe the activities shown in the selector.

    Parameters
    ----------
    activities : pandas.DataFrame
        Summary rows with the ``id``, ``name``, ``date`` (French Strava
        format) and ``distance_m`` columns.

    Returns
    -------
    list of ActivitySummary
        One entry per row, in the order of `activities`.
    """
    dates = pd.to_datetime(activities["date"].map(parse_strava_date))
    distance_km = pd.to_numeric(activities["distance_m"], errors="coerce") / (
        METRES_PER_KM
    )
    rows = pd.DataFrame(
        {
            "id": activities["id"].astype(int),
            "name": activities["name"].fillna("").astype(str),
            "date": _none_if_missing(dates.dt.strftime(ISO_DATETIME_FORMAT)),
            "distance_km": _none_if_missing(distance_km),
        }
    )
    return [ActivitySummary.model_validate(row) for row in rows.to_dict("records")]


def metrics_view(metrics: ActivityMetrics) -> dict[str, float | None]:
    """Convert the metrics to JSON-compatible values.

    Parameters
    ----------
    metrics : ActivityMetrics
        Metrics of an activity.

    Returns
    -------
    dict[str, float or None]
        Each field of `metrics`, NaN replaced by None (not valid JSON).
    """
    return {name: _finite_or_none(value) for name, value in asdict(metrics).items()}


def charts_view(plot: PlotActivities, x_axis: XAxis = "distance") -> ChartsView:
    """Build the metrics and every chart of the activity page.

    A chart that cannot be built is reported with its error instead of
    failing the whole view, as the other charts are still useful.

    Parameters
    ----------
    plot : PlotActivities
        Activity to show.
    x_axis : {"distance", "time"}, optional
        Abscissa of the heart-rate chart. Default is ``"distance"``.

    Returns
    -------
    ChartsView
        Metrics and charts of the activity.
    """
    charts = {
        name: build_chart(name, partial(build, plot, x_axis))
        for name, build in CHARTS.items()
    }
    return ChartsView(
        metrics=metrics_view(compute_activity_metrics(plot.data)), charts=charts
    )


def track_view(
    plot: PlotActivities,
    color_by: ColorBy = "altitude",
    texture: MapTexture = "osm",
    show_km_markers: bool = True,
) -> TrackView:
    """Build the 3D map of the GPS track.

    Parameters
    ----------
    plot : PlotActivities
        Activity to map.
    color_by : {"altitude", "heart_rate", "pace"}, optional
        Variable colouring the track. Default is ``"altitude"``.
    texture : {"osm", "satellite"}, optional
        Imagery draped over the terrain. Default is ``"osm"``.
    show_km_markers : bool, optional
        Add a marker at each kilometre. Default is True.

    Returns
    -------
    TrackView
        deck.gl JSON and colour legend.

    Raises
    ------
    ValueError
        If the activity has no valid GPS coordinates.
    """
    track = plot.track_map(
        color_by=color_by, show_km_markers=show_km_markers, texture=texture
    )
    return TrackView(
        deck=json.loads(track.deck.to_json()),
        legend=TrackLegendView(**asdict(track.legend)),
    )


def _load_from_computed_file(
    file_name: str, segment_width: int, progress_cb: ProgressCallback
) -> RunningActivities:
    return RunningActivities.from_computed_file(
        file_name, segment_width=segment_width, progress_cb=progress_cb
    )


class ActivityService:
    """Catalogue of the running activities and cache of the loaded ones.

    Parameters
    ----------
    summary_loader : callable, optional
        Returns the Strava activity summary. Default is `load_activities`.
    activity_loader : callable, optional
        Loads ``(file_name, segment_width, progress_cb)`` into a
        `RunningActivities`. Default reads the cached CSV, or the FIT file.
    max_loaded : int, optional
        Number of segmented activities kept in memory, least recently used
        first evicted. Default is ``MAX_LOADED_ACTIVITIES``.

    Notes
    -----
    The methods are thread-safe: the loading runs in the job threads while
    the routes read the cache.
    """

    def __init__(
        self,
        summary_loader: SummaryLoader = load_activities,
        activity_loader: ActivityLoader = _load_from_computed_file,
        max_loaded: int = MAX_LOADED_ACTIVITIES,
    ) -> None:
        self._summary_loader = summary_loader
        self._activity_loader = activity_loader
        self._max_loaded = max_loaded
        self._summary: pd.DataFrame | None = None
        self._loaded: OrderedDict[tuple[int, int], RunningActivities] = OrderedDict()
        self._lock = threading.Lock()

    def running_activities(self) -> pd.DataFrame:
        """Return the running activities of the summary, read once.

        Returns
        -------
        pandas.DataFrame
            Summary rows of the ``run`` activities.

        Raises
        ------
        FileNotFoundError
            If the summary file does not exist; retried at the next call.
        """
        with self._lock:
            if self._summary is None:
                self._summary = running_activities(self._summary_loader())
            return self._summary

    def summaries(self) -> list[ActivitySummary]:
        """Describe the running activities, longest first.

        Returns
        -------
        list of ActivitySummary
            Entries of the activity selector.
        """
        activities = self.running_activities().sort_values(
            "distance_m", ascending=False, na_position="last"
        )
        return summarize_activities(activities)

    def file_name(self, activity_id: int) -> str:
        """Return the FIT file of a running activity.

        Parameters
        ----------
        activity_id : int
            Strava identifier.

        Returns
        -------
        str
            File name relative to the data directory.

        Raises
        ------
        UnknownActivityError
            If `activity_id` is not a running activity of the summary.
        """
        return activity_file_name(self.running_activities(), activity_id)

    def _activity_name(self, activity_id: int) -> str:
        """Return the summary name of an activity, ``""`` if missing."""
        activities = self.running_activities()
        names = activities.loc[activities["id"].eq(activity_id), "name"].dropna()
        return "" if names.empty else str(names.iloc[0])

    def load(
        self,
        activity_id: int,
        segment_width: int,
        progress_cb: ProgressCallback,
    ) -> None:
        """Load and cache an activity, unless it is already cached.

        Parameters
        ----------
        activity_id : int
            Strava identifier.
        segment_width : int
            Segment length, in metres.
        progress_cb : ProgressCallback
            Progress reporter of the loading.

        Raises
        ------
        UnknownActivityError
            If `activity_id` is not a running activity of the summary.
        """
        key = (activity_id, segment_width)
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return
        # loaded outside the lock: the other activities stay readable
        activity = self._activity_loader(
            self.file_name(activity_id), segment_width, progress_cb
        )
        # the cached CSV has no name: take it from the summary (chart subtitles)
        activity.name = self._activity_name(activity_id)
        with self._lock:
            self._loaded[key] = activity
            self._loaded.move_to_end(key)
            if len(self._loaded) > self._max_loaded:
                evicted, _ = self._loaded.popitem(last=False)
                logger.info("Evicted activity %s from the cache", evicted)

    def plot(
        self, activity_id: int, segment_width: int, drop_threshold: float
    ) -> PlotActivities:
        """Return the plotting interface of a loaded activity.

        Parameters
        ----------
        activity_id : int
            Strava identifier.
        segment_width : int
            Segment length, in metres.
        drop_threshold : float
            Altitude change, in metres, above which a segment is uphill.

        Returns
        -------
        PlotActivities
            New plotting interface; the cached activity is not modified.

        Raises
        ------
        ActivityNotLoadedError
            If the activity was not loaded with this width.
        """
        key = (activity_id, segment_width)
        with self._lock:
            activity = self._loaded.get(key)
            if activity is not None:
                self._loaded.move_to_end(key)
        if activity is None:
            raise ActivityNotLoadedError(f"Activity {key} is not loaded")
        return PlotActivities(activity, drop_threshold)
