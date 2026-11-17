"""Plotly figures for a single running activity."""

from collections.abc import Callable, Iterator
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from TrailPredictor.activities import RunningActivities
from TrailPredictor.features import HEART_RATE_ZONES, HearthRateZone, build_features
from TrailPredictor.plot._common import TARGETS_UNITS, _base_layout
from TrailPredictor.speed_columns import format_pace

DropCategory = Literal["Uphill", "Downhill", "Flat"]
CATEGORY_COLORS = {"Uphill": "#0072B2", "Downhill": "#D55E00", "Flat": "#777777"}

PAUSE_COLOR = "#004E00"
PAUSE_MARKER_SIZE = 7


def _pace_text(pace: float) -> str:
    return format_pace(pace).replace(" min/km", "/km").replace("--:--/km", "—")


def _pace_ticks(values: pd.Series) -> tuple[list[float], list[str]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return [], []
    low = np.floor(numeric.min() * 2) / 2
    high = np.ceil(numeric.max() * 2) / 2
    ticks = np.arange(low, high + 0.25, 0.5)
    return ticks.tolist(), [_pace_text(value) for value in ticks]


class PlotActivities(RunningActivities):
    """Plotting interface of a single activity.

    Each segment is labelled with a drop category (uphill, downhill, flat)
    and a heart-rate zone.

    Parameters
    ----------
    running_activities : RunningActivities
        Activity to plot; its data and prediction are copied.
    drop_threshold : float, optional
        Altitude change in metres above which a segment is uphill (below its
        opposite, downhill). Default is 2.0.

    See Also
    --------
    PlotDataset : Plots of the whole dataset.
    """

    def __init__(
        self, running_activities: RunningActivities, drop_threshold: float = 2.0
    ):
        super().__init__(
            running_activities.data, prediction=running_activities._prediction
        )
        self._name = running_activities.name
        self._data = build_features(self._data, drop_threshold=drop_threshold)

    @property
    def target_label(self) -> str:
        """Return the target name with its unit."""
        return f"{self.target_name} ({TARGETS_UNITS[self.target_name]})"

    @property
    def target_unit(self) -> str:
        """Return the unit of the target."""
        return TARGETS_UNITS[self.target_name]

    @property
    def iter_drop_categories(
        self,
    ) -> Iterator[tuple[DropCategory, npt.NDArray[np.intp], pd.DataFrame]]:
        """Iterate over the contiguous runs of each drop category.

        Each run is extended by one segment on each side so that the drawn
        lines join; runs shorter than two segments are skipped.

        Yields
        ------
        category : DropCategory
            Drop category of the run.
        idx : numpy.ndarray
            Positional indices of the segments of the run.
        segment : pandas.DataFrame
            Rows of the run.
        """
        n = len(self._data)

        for category in get_args(DropCategory):
            indexes = np.flatnonzero(self._data["drop_category"] == category)
            if indexes.size == 0:
                continue

            for segment in np.split(indexes, np.where(np.diff(indexes) != 1)[0] + 1):
                # Extend the segment by one point on each side (overlap with neighbors)
                start = max(segment[0] - 1, 0)
                stop = min(segment[-1] + 1, n - 1)
                idx = np.arange(start, stop + 1)
                if len(idx) < 2:
                    continue

                yield (category, idx, self._data.iloc[idx])

    def iter_heart_rate_zones(
        self,
    ) -> Iterator[tuple[HearthRateZone, npt.NDArray[np.intp], pd.DataFrame]]:
        """Iterate over the contiguous runs of each heart-rate zone.

        Each run is extended by one segment on each side so that the drawn
        lines join; runs shorter than two segments are skipped.

        Yields
        ------
        zone : HearthRateZone
            Heart-rate zone of the run.
        idx : numpy.ndarray
            Positional indices of the segments of the run.
        segment : pandas.DataFrame
            Rows of the run.
        """
        for zone in HEART_RATE_ZONES:
            indexes = np.flatnonzero(self._data["heart_rate_zone"] == zone)
            if indexes.size == 0:
                continue

            for segment in np.split(indexes, np.where(np.diff(indexes) != 1)[0] + 1):
                # Extend the segment by one point on each side (overlap with neighbors)
                start = max(segment[0] - 1, 0)
                stop = min(segment[-1] + 1, len(self._data) - 1)
                idx = np.arange(start, stop + 1)
                if len(idx) < 2:
                    continue

                yield zone, idx, self._data.iloc[idx]

    def _add_pause_markers(
        self,
        fig: go.Figure,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        trace_type: Callable[..., go.Scatter] = go.Scatter,
        **add_trace_kwargs: Any,
    ) -> None:
        """Overlay small markers on the segments flagged as pauses.

        Nothing is added when the activity has no pause.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to which the markers are added (modified in place).
        x, y : array-like of shape (n,)
            Coordinates of every segment, aligned with the activity data;
            only the paused segments are drawn.
        trace_type : callable, optional
            Plotly trace class; ``go.Scatter`` by default. Map traces take
            ``lat``/``lon`` instead of ``x``/``y``.
        **add_trace_kwargs
            Forwarded to ``fig.add_trace`` (e.g. ``row``, ``col``,
            ``secondary_y``).
        """
        is_pause = self._data["pause"].to_numpy(dtype=bool)
        if not is_pause.any():
            return
        x_pause = np.asarray(x)[is_pause]
        y_pause = np.asarray(y)[is_pause]
        is_map = trace_type is not go.Scatter
        coordinates = (
            {"lat": x_pause, "lon": y_pause} if is_map else {"x": x_pause, "y": y_pause}
        )
        fig.add_trace(
            trace_type(
                **coordinates,
                mode="markers",
                marker={"color": PAUSE_COLOR, "size": PAUSE_MARKER_SIZE},
                name="Pause",
                legendgroup="pause",
                hoverinfo="skip" if is_map else None,
                hovertemplate=None if is_map else "Pause<extra></extra>",
            ),
            **add_trace_kwargs,
        )

    # ===================================================================================== #
    #                                   ALTITUDE PLOT
    # ===================================================================================== #
    def altitude(self) -> go.Figure:
        """Plot the altitude profile coloured by drop category.

        Returns
        -------
        plotly.graph_objects.Figure
            Filled altitude profile against the distance in km.
        """
        fig = _base_layout(go.Figure(), "Altitude Profile", self.name)

        show_legend: dict[DropCategory, bool] = {
            "Uphill": True,
            "Downhill": True,
            "Flat": True,
        }
        for category, _, segment in self.iter_drop_categories:
            fig.add_trace(
                go.Scatter(
                    x=segment["distance"] / 1000,
                    y=segment["altitude"],
                    mode="lines",
                    fill="tozeroy",
                    line={"color": CATEGORY_COLORS[category], "width": 1.5},
                    fillcolor=CATEGORY_COLORS[category],
                    name=category,
                    legendgroup=category,
                    showlegend=show_legend[category],
                    customdata=segment[["distance", "altitude", "altitude_drop"]],
                    hovertemplate=(
                        "Distance : %{customdata[0]:.0f} m<br>"
                        "Altitude : %{customdata[1]:.1f} m<br>"
                        "Variation : %{customdata[2]:+.1f} m<extra></extra>"
                    ),
                )
            )
            show_legend[category] = False

        self._add_pause_markers(fig, self.distance_km, self._data["altitude"])
        fig.update_xaxes(title="Distance (km)")
        fig.update_yaxes(title="Altitude (m)")
        return fig

    # =================================================================================== #
    #                                   GPS TRACK PLOT
    # =================================================================================== #
    def track_map(  # pylint: disable=too-many-locals
        self,
        color_by: Literal["altitude", "heart_rate", "pace"] = "altitude",
        show_km_markers: bool = True,
    ) -> go.Figure:
        """Plot the GPS track on an OpenStreetMap background.

        Parameters
        ----------
        color_by : {"altitude", "heart_rate", "pace"}, optional
            Variable used to colour the track. Default is ``"altitude"``.
        show_km_markers : bool, optional
            Add a marker at each kilometre. Default is True.

        Returns
        -------
        plotly.graph_objects.Figure
            Map of the track.

        Raises
        ------
        ValueError
            If `color_by` is invalid or the activity has no valid GPS
            coordinates.
        """
        data = self._data.copy()
        latitude = data["position_lat"]
        longitude = data["position_long"]
        distance = data["distance"]

        if color_by == "altitude":
            values = data["altitude"]
            label = "Altitude (m)"
        elif color_by == "heart_rate":
            values = data["heart_rate"]
            label = "Heart Rate (bpm)"
        elif color_by == "pace":
            values = data["pace"]
            label = "Pace (min/km)"
        else:
            raise ValueError("color_by must be 'altitude', 'heart_rate' or 'pace'")

        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        cmin, cmax = finite.quantile([0.05, 0.95]) if not finite.empty else (0, 1)
        colorscale = "RdYlGn_r" if color_by == "pace" else "Viridis"
        map_trace = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox
        trace = {
            "lat": latitude,
            "lon": longitude,
            "mode": "lines+markers",
            "line": {"color": "#666666", "width": 2},
            "marker": {
                "color": values,
                "colorscale": colorscale,
                "cmin": cmin,
                "cmax": cmax,
                "size": 5,
                "colorbar": {"title": label},
            },
            "customdata": np.column_stack((distance, values)),
            "hovertemplate": (
                "Distance : %{customdata[0]:.0f} m<br>"
                f"{label} : %{{customdata[1]:.2f}}<extra></extra>"
            ),
            "name": "Parcours",
        }
        if latitude.notna().sum() < 2 or longitude.notna().sum() < 2:
            raise ValueError("L'activité ne contient pas de coordonnées GPS valides.")
        fig = _base_layout(go.Figure(map_trace(**trace)), "Parcours GPS", self.name)
        map_options = {
            "style": "open-street-map",
            "center": {"lat": latitude.mean(), "lon": longitude.mean()},
            "zoom": self._map_zoom(latitude, longitude),
        }
        if hasattr(go, "Scattermap"):
            fig.update_layout(map=map_options)
        else:
            fig.update_layout(mapbox=map_options)

        fig.add_trace(
            map_trace(
                lat=[latitude.iloc[0]],
                lon=[longitude.iloc[0]],
                mode="markers",
                marker={"color": "#009E73", "size": 11},
                name="Départ",
            )
        )
        fig.add_trace(
            map_trace(
                lat=[latitude.iloc[-1]],
                lon=[longitude.iloc[-1]],
                mode="markers",
                marker={"color": "#D55E00", "size": 11},
                name="Arrivée",
            )
        )
        self._add_pause_markers(fig, latitude, longitude, trace_type=map_trace)
        if show_km_markers:
            targets = np.arange(1000, distance.max(), 1000)
            positions = np.asarray(
                [np.abs(distance.to_numpy() - target).argmin() for target in targets],
                dtype=int,
            )
            if len(positions):
                fig.add_trace(
                    map_trace(
                        lat=latitude.iloc[positions],
                        lon=longitude.iloc[positions],
                        mode="markers+text",
                        text=[f"km {index}" for index in range(1, len(positions) + 1)],
                        textposition="top center",
                        marker={"color": "#333333", "size": 7},
                        name="Kilomètres",
                    )
                )
        return fig

    @staticmethod
    def _map_zoom(latitude: pd.Series, longitude: pd.Series) -> float:
        span = max(
            float(latitude.max() - latitude.min()),
            float(longitude.max() - longitude.min()),
        )
        return float(np.clip(11 - np.log2(max(span, 1e-5) / 0.05), 3, 15))

    # ================================================================================= #
    #                                   PACE PLOT
    # ================================================================================= #
    def pace_by_segment(self) -> go.Figure:
        """Plot the pace of each segment.

        Returns
        -------
        plotly.graph_objects.Figure
            Bar chart of the pace, each bar spanning the distance covered by
            its segment.
        """
        data = self._data.copy()
        distance = data["distance"]
        raw_pace = pd.to_numeric(self._data["pace"], errors="coerce")
        pace = raw_pace.copy()
        # set to NA the pace bigger than 20 min/km
        pace[pace > 20] = np.nan
        average = pace.mean()

        difference = (pace - average) * 60
        # Compress extreme differences while preserving whether the segment is
        # faster or slower than the average.
        color_difference = np.sign(difference) * np.log1p(np.abs(difference))
        _segment_width = np.repeat(self._segment_width, len(data))

        fig = _base_layout(go.Figure(), "Allure par segment", self.name)
        fig.add_trace(
            go.Bar(
                x=distance / 1000,
                y=pace,
                width=_segment_width / 1000,
                marker={
                    "color": color_difference,
                    "colorscale": "RdYlGn_r",
                    "cmid": 0,
                    "colorbar": {"title": "Écart transformé"},
                },
                customdata=np.column_stack(
                    (np.arange(1, len(data) + 1), difference, _segment_width)
                ),
                hovertemplate=(
                    "Segment %{customdata[0]:.0f}<br>Allure : %{y:.2f} min/km<br>"
                    "Écart : %{customdata[1]:+.0f} s/km<br>"
                    "Distance : %{customdata[2]:.0f} m<extra></extra>"
                ),
                name="Segments",
            )
        )
        fig.add_hline(
            y=average,
            line_dash="dot",
            annotation_text=f"Moyenne : {_pace_text(average)}",
        )
        # Pauses are usually slower than the displayed range: pin them to its edge.
        self._add_pause_markers(fig, distance / 1000, raw_pace.clip(upper=pace.max()))
        ticks, labels = _pace_ticks(pace)
        fig.update_xaxes(title="Distance (km)")

        fig.update_yaxes(
            title="Allure (min/km)",
            autorange="reversed",
            tickvals=ticks,
            ticktext=labels,
        )
        return fig

    # =================================================================================== #
    #                                   HEART RATE PLOT
    # =================================================================================== #
    def hr_overlay(self, x_axis: Literal["distance", "time"] = "distance") -> go.Figure:
        """Overlay heart rate and altitude on two y axes.

        Parameters
        ----------
        x_axis : {"distance", "time"}, optional
            Quantity on the x axis. Default is ``"distance"``.

        Returns
        -------
        plotly.graph_objects.Figure
            Heart rate (left axis) and altitude (right axis).

        Raises
        ------
        ValueError
            If `x_axis` is invalid.
        """
        if x_axis not in ("distance", "time"):
            raise ValueError("x_axis must be 'distance' or 'time'")
        data = self._data.copy()
        x = (
            self.distance_km
            if x_axis == "distance"
            else data["elapsed_time"].dt.total_seconds() / 60
        )
        fig = _base_layout(
            make_subplots(specs=[[{"secondary_y": True}]]),
            "Fréquence cardiaque et altitude",
            self.name,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data["altitude"],
                mode="lines",
                fill="tozeroy",
                name="Altitude",
                line={"color": "#56B4E9"},
                hovertemplate="Altitude : %{y:.1f} m<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data["heart_rate"],
                mode="lines",
                name="Fréquence cardiaque",
                line={"color": "#CC79A7"},
                hovertemplate="FC : %{y:.0f} bpm<extra></extra>",
            ),
            secondary_y=True,
        )
        self._add_pause_markers(fig, x, data["heart_rate"], secondary_y=True)
        fig.update_xaxes(
            title="Distance (km)" if x_axis == "distance" else "Time (min)"
        )
        fig.update_yaxes(title="Heart rate (bpm)", secondary_y=False)
        fig.update_yaxes(title="Altitude (m)", secondary_y=True)
        return fig

    # ================================================================================== #
    #                                   SPEED PLOT
    # ================================================================================== #
    def speed(self) -> go.Figure:
        """Overlay speed and altitude on two y axes.

        Returns
        -------
        plotly.graph_objects.Figure
            True speed, predicted speed when a prediction is available (left
            axis) and altitude (right axis).

        Raises
        ------
        ValueError
            If reading the prediction fails for another reason than a
            missing prediction.
        """
        data = self._data.copy()
        x = self.distance_km
        fig = _base_layout(
            make_subplots(specs=[[{"secondary_y": True}]]),
            "Vitesse et altitude",
            self.name,
        )

        customdata = self.target.to_numpy().reshape(-1, 1)

        hovertemplate = (
            "Distance : %{x:.2f} km<br>"
            "Altitude : %{y:.1f} m<br>"
            f"Target : %{{customdata[0]:.1f}} {self.target_unit}"
            "<extra></extra>"
        )

        # try to use prediction or catch ValueError("No prediction available, prediction is None.")
        try:
            if self.prediction is not None:
                customdata = np.column_stack(
                    [
                        self.target.to_numpy(),
                        self.prediction,
                        np.abs(self.prediction - self.target.to_numpy()),
                    ]
                )

                hovertemplate = (
                    "Distance : %{x:.2f} km<br>"
                    "Altitude : %{y:.1f} m<br>"
                    f"Target : %{{customdata[0]:.1f}} {self.target_unit}<br>"
                    f"Prediction : %{{customdata[1]:.1f}} {self.target_unit}<br>"
                    f"Erreur : %{{customdata[2]:.1f}} {self.target_unit}"
                    "<extra></extra>"
                )

        except ValueError as e:
            if str(e) != "No prediction available, prediction is None.":
                raise e

        fig.add_trace(
            go.Scatter(
                x=x,
                y=data["altitude"],
                mode="lines",
                fill="tozeroy",
                name="Altitude",
                line={"color": "#56B4E9"},
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=self.target,
                mode="lines",
                name="Target",
                line={"color": "#44AF52"},
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            secondary_y=True,
        )
        if self.prediction is not None:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=self.prediction,
                    mode="lines",
                    name="Target predicted",
                    line={"color": "#8256E9"},
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                secondary_y=True,
            )

        self._add_pause_markers(fig, x, self.target, secondary_y=True)
        fig.update_xaxes(title="Distance (km)")
        fig.update_yaxes(title="Altitude (m)", secondary_y=False)
        fig.update_yaxes(title=self.target_label, secondary_y=True)
        return fig

    # ================================================================================== #
    #                                   PREDICTION PLOT
    # ================================================================================== #
    def error_prediction(self) -> go.Figure:
        """Plot the predicted vs true speed and the error histogram.

        Returns
        -------
        plotly.graph_objects.Figure
            Two rows: predicted and true speed, then the histogram of the
            prediction error.

        Raises
        ------
        ValueError
            If no prediction is available.
        """
        if self.prediction is None:
            raise ValueError("No prediction available")

        fig = _base_layout(
            make_subplots(rows=2, cols=1), "Prediction vs True Target", self.name
        )
        fig.add_trace(
            go.Scatter(
                x=self.distance_km,
                y=self.target,
                mode="lines",
                name="True target",
                line={"color": "#44AF52"},
                hovertemplate="Distance : %{x:.2f} km<br>True target : %{y:.1f} m/s<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.distance_km,
                y=self.prediction,
                mode="lines",
                name="Predicted target",
                line={"color": "#8256E9"},
                hovertemplate=(
                    "Distance : %{x:.2f} km<br>"
                    "Predicted target : %{y:.1f} m/s<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        self._add_pause_markers(fig, self.distance_km, self.target, row=1, col=1)
        error = self.prediction - self.target
        fig.add_trace(
            go.Histogram(
                x=error,
                nbinsx=50,
                marker={"color": "#E95656"},
                name="Error histogram",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title="Distance (km)", row=1, col=1)
        fig.update_yaxes(title=self.target_label, row=1, col=1)
        fig.update_xaxes(title=f"Error ({self.target_unit})", row=2, col=1)
        fig.update_yaxes(title="Count", row=2, col=1)
        return fig

    # ================================================================================== #
    #                                   TIME IN HEART RATE ZONES PLOT
    # ================================================================================== #
    def time_in_hr_zones(self) -> go.Figure:
        """Plot the time spent in each heart-rate zone.

        Returns
        -------
        plotly.graph_objects.Figure
            Bar chart of the duration per zone.
        """
        durations = {
            zone: self._data.loc[self._data["heart_rate_zone"].eq(zone), "elapsed_time"]
            .dt.total_seconds()
            .sum()
            for zone in HEART_RATE_ZONES
        }

        total = sum(durations.values())
        values = np.array(list(durations.values()))
        fig = _base_layout(
            go.Figure(), "Temps passé dans les zones cardiaques", self.name
        )
        fig.add_trace(
            go.Bar(
                y=list(durations),
                x=values / 60,
                orientation="h",
                text=[
                    f"{value / total * 100:.1f} %" if total else "0.0 %"
                    for value in values
                ],
                textposition="auto",
                customdata=values,
                hovertemplate="%{y}<br>Durée : %{customdata:.0f} s<extra></extra>",
                name="Durée",
            )
        )
        fig.update_xaxes(title="Temps (minutes)")
        return fig
