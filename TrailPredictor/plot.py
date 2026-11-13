"""Plotly figures for the dataset, the activities and the Minetti curves."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from TrailPredictor.activities import RunningActivities
from TrailPredictor.speed_columns import TARGETS
from TrailPredictor.dataset import Dataset
from TrailPredictor.minetti import cost_of_running, cost_of_walking
from TrailPredictor.speed_columns import format_pace

DropCategory = Literal["Uphill", "Downhill", "Flat"]
CATEGORY_COLORS = {"Uphill": "#0072B2", "Downhill": "#D55E00", "Flat": "#777777"}

HearthRateZoneLabel = Literal["Endurance", "Seuil", "Haut", "Hors zones"]
HearthRateZone = {
    "Endurance": (120, 150),
    "Seuil": (150, 170),
    "Haut": (170, 210),
    "Hors zones": (0, 120),
}


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin={"l": 50, "r": 25, "t": 55, "b": 45},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hoverlabel={"namelength": -1},
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a hexadecimal color to an rgba CSS color."""
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


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


TARGETS_UNITS = {
    "pace": "min/km",
    "log_pace": "log(min/km)",
    "speed": "m/s",
    "log_speed": "log(m/s)",
    "accel_pace": "min/km²",
    "log_accel_pace": "log(min/km²)",
    "accel_speed": "m/s²",
    "log_accel_speed": "log(m/s²)",
}
TARGETS_COLORS = {
    "pace": "#56B4E9",
    "log_pace": "#E95656",
    "speed": "#5BE956",
    "log_speed": "#E456E9",
    "accel_pace": "#E9E956",
    "log_accel_pace": "#56E9E9",
    "accel_speed": "#E9A956",
    "log_accel_speed": "#A956E9",
}


class PlotDataset(Dataset):
    """Plotting interface of a dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to plot; its data and target are copied.

    See Also
    --------
    PlotActivities : Plots of a single activity.
    """

    def __init__(self, dataset: Dataset):
        super().__init__(dataset._data, target=dataset.target_name)

    @property
    def target_label(self) -> str:
        """Return the target name with its unit."""
        return f"{self.target_name} ({TARGETS_UNITS[self.target_name]})"

    @property
    def target_unit(self) -> str:
        """Return the unit of the target."""
        return TARGETS_UNITS[self.target_name]

    def merge_figure(
        self, plot_fct: Callable[[], go.Figure], title: str, **kwargs
    ) -> go.Figure:
        """Draw the same figure for every target on a 3 x 3 grid.

        Parameters
        ----------
        plot_fct : Callable[..., plotly.graph_objects.Figure]
            Bound plotting method (e.g. ``self.target_histogram``), called
            once per target with `kwargs`.
        title : str
            Title of the merged figure.
        **kwargs
            Extra keyword arguments passed to `plot_fct`.

        Returns
        -------
        plotly.graph_objects.Figure
            One subplot per target; the current target is restored.
        """
        merged = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=TARGETS,
            vertical_spacing=0.15,
        )
        current_target_name = self.target_name
        figs = {}
        try:
            for col in TARGETS:
                self.target = col
                figs[col] = plot_fct(**kwargs)
        finally:
            self.target = current_target_name

        for i, fig in enumerate(figs.values(), start=1):
            row = (i - 1) // 3 + 1
            col = (i - 1) % 3 + 1
            x_title = fig.layout.xaxis.title.text
            y_title = fig.layout.yaxis.title.text
            merged.update_xaxes(title_text=x_title, row=row, col=col)
            merged.update_yaxes(title_text=y_title, row=row, col=col)
            for trace in fig.data:
                trace.showlegend = i == 1
                merged.add_trace(trace, row=row, col=col)

        _base_layout(merged, title=title)
        # put legend below the plots
        merged.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
            )
        )
        return merged

    # ================================================================================= #
    #                             TARGET HISTOGRAM PLOT
    # ================================================================================= #
    def target_histogram(self) -> go.Figure:
        """Plot the histogram of the target.

        Returns
        -------
        plotly.graph_objects.Figure
            Histogram with 50 bins.
        """
        fig = _base_layout(go.Figure(), f"Histogram of {self.target_name}")
        fig.add_trace(
            go.Histogram(
                x=self.target,
                nbinsx=50,
                marker={"color": TARGETS_COLORS[self.target_name]},
                name=self.target_label,
            )
        )
        fig.update_xaxes(title=self.target_name)
        fig.update_yaxes(title="Count")
        return fig

    # ================================================================================= #
    #                             TARGET VS FEATURE PLOT
    # ================================================================================= #
    def target_vs_feat(self, features="altitude_drop") -> go.Figure:
        """Plot the target against a feature, one point per segment.

        Parameters
        ----------
        features : str, optional
            Name of a numeric column. Default is ``"altitude_drop"``.

        Returns
        -------
        plotly.graph_objects.Figure
            Scatter plot of the target vs the feature.

        Raises
        ------
        ValueError
            If the feature is missing, not numeric or only missing values.
        """
        if features not in self._data.columns:
            raise ValueError(f"Invalid feature: {features}")
        values = self._data[features]
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"Feature '{features}' must be numeric")
        if values.dropna().empty:
            raise ValueError(f"Feature '{features}' contains no numeric values")
        fig = _base_layout(go.Figure(), f"{self.target_name} vs {features} per segment")
        fig.add_trace(
            go.Scatter(
                x=values,
                y=self.target,
                mode="markers",
                marker={"color": TARGETS_COLORS[self.target_name], "size": 5},
                name=self.target_label,
            )
        )
        fig.update_xaxes(title=features)
        fig.update_yaxes(title=self.target_label)
        return fig

    # ================================================================================= #
    #                                 VS PREDICTION PLOT
    # ================================================================================= #
    def vs_prediction(self) -> go.Figure:
        """Plot the predictions against the true values, and the residuals.

        Returns
        -------
        plotly.graph_objects.Figure
            Two subplots: predicted vs true values with the unity line, and
            residuals vs predicted values.

        Raises
        ------
        ValueError
            If no prediction is set.
        """
        fig = _base_layout(
            make_subplots(
                rows=1, cols=2, subplot_titles=["Target vs Prediction", "Residuals"]
            ),
            "Prediction vs True Values",
        )
        fig.add_trace(
            go.Scatter(
                x=self.target,
                y=self.prediction,
                mode="markers",
                marker={"color": TARGETS_COLORS[self.target_name], "size": 5},
                name=self.target_label,
                hovertemplate=f"Prediction : %{{x:.2f}}<br>True value : %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # add unity line
        fig.add_trace(
            go.Line(
                x=[self.prediction.min(), self.prediction.max()],
                y=[self.prediction.min(), self.prediction.max()],
                mode="lines",
                line={"color": "#000000", "dash": "dash"},
                name="Unity line",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        residuals = self.target - self.prediction
        fig.add_trace(
            go.Scatter(
                x=self.prediction,
                y=residuals,
                mode="markers",
                marker={"color": TARGETS_COLORS[self.target_name], "size": 5},
                name="Residuals",
                hovertemplate=f"Prediction : %{{x:.2f}}<br>Residual : %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title=f"True value ({TARGETS_UNITS[self.target_name]})", row=1, col=1
        )
        fig.update_yaxes(
            title=f"Prediction ({TARGETS_UNITS[self.target_name]})", row=1, col=1
        )
        fig.update_xaxes(
            title=f"Prediction ({TARGETS_UNITS[self.target_name]})", row=1, col=2
        )
        fig.update_yaxes(
            title=f"Residual ({TARGETS_UNITS[self.target_name]})", row=1, col=2
        )
        return fig


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
        self._data = self._data[
            [
                "timestamp",
                "elapsed_time",
                "position_lat",
                "position_long",
                "distance",
                "altitude",
                "heart_rate",
            ]
            + list(TARGETS)
        ]

        self._data["altitude_drop"] = self._data["altitude"].diff()

        self._data["drop_category"] = np.select(
            [
                self._data["altitude_drop"] > drop_threshold,
                self._data["altitude_drop"] < -drop_threshold,
            ],
            ["Uphill", "Downhill"],
            default="Flat",
        )

        self._data["heart_rate_zone"] = np.select(
            [
                self._data["heart_rate"].between(z[0], z[1], inclusive="left")
                for z in HearthRateZone.values()
            ],
            list(HearthRateZone.keys()),
            default="Hors zones",
        )

    @property
    def target_label(self) -> str:
        """Return the target name with its unit."""
        return f"{self.target_name} ({TARGETS_UNITS[self.target_name]})"

    @property
    def target_unit(self) -> str:
        """Return the unit of the target."""
        return TARGETS_UNITS[self.target_name]

    @property
    def iter_drop_categories(self):
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

        for category in DropCategory.__args__:
            indexes = np.flatnonzero(self._data["drop_category"] == category)
            if not len(indexes):
                continue

            for segment in np.split(indexes, np.where(np.diff(indexes) != 1)[0] + 1):
                # Extend the segment by one point on each side (overlap with neighbors)
                start = max(segment[0] - 1, 0)
                stop = min(segment[-1] + 1, n - 1)
                idx: np.ndarray = np.arange(start, stop + 1)
                if len(idx) < 2:
                    continue

                yield (category, idx, self._data.iloc[idx])

    def iter_heart_rate_zones(self):
        """Iterate over the contiguous runs of each heart-rate zone.

        Each run is extended by one segment on each side so that the drawn
        lines join; runs shorter than two segments are skipped.

        Yields
        ------
        zone : HearthRateZoneLabel
            Heart-rate zone of the run.
        idx : numpy.ndarray
            Positional indices of the segments of the run.
        segment : pandas.DataFrame
            Rows of the run.
        """
        for zone in HearthRateZoneLabel.__args__:
            indexes = np.flatnonzero(self._data["heart_rate_zone"] == zone)
            if not len(indexes):
                continue

            for segment in np.split(indexes, np.where(np.diff(indexes) != 1)[0] + 1):
                # Extend the segment by one point on each side (overlap with neighbors)
                start = max(segment[0] - 1, 0)
                stop = min(segment[-1] + 1, len(self._data) - 1)
                idx: np.ndarray = np.arange(start, stop + 1)
                if len(idx) < 2:
                    continue

                yield zone, idx, self._data.iloc[idx]

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
        fig = _base_layout(go.Figure(), "Altitude Profile")

        show_legend: dict[DropCategory, bool] = {
            "Uphill": True,
            "Downhill": True,
            "Flat": True,
        }
        for i, (category, _, segment) in enumerate(self.iter_drop_categories):
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

        fig.update_xaxes(title="Distance (km)")
        fig.update_yaxes(title="Altitude (m)")
        return fig

    # =================================================================================== #
    #                                   GPS TRACK PLOT
    # =================================================================================== #
    def track_map(
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
        trace = dict(
            lat=latitude,
            lon=longitude,
            mode="lines+markers",
            line={"color": "#666666", "width": 2},
            marker={
                "color": values,
                "colorscale": colorscale,
                "cmin": cmin,
                "cmax": cmax,
                "size": 5,
                "colorbar": {"title": label},
            },
            customdata=np.column_stack((distance, values)),
            hovertemplate=(
                "Distance : %{customdata[0]:.0f} m<br>"
                f"{label} : %{{customdata[1]:.2f}}<extra></extra>"
            ),
            name="Parcours",
        )
        if latitude.notna().sum() < 2 or longitude.notna().sum() < 2:
            raise ValueError("L'activité ne contient pas de coordonnées GPS valides.")
        fig = _base_layout(go.Figure(map_trace(**trace)), "Parcours GPS")
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
        pace = pd.to_numeric(self._data["pace"], errors="coerce")
        # set to NA the pace bigger than 20 min/km
        pace[pace > 20] = np.nan
        average = pace.mean()

        difference = (pace - average) * 60
        # Compress extreme differences while preserving whether the segment is
        # faster or slower than the average.
        color_difference = np.sign(difference) * np.log1p(np.abs(difference))
        _segment_width = np.repeat(self._segment_width, len(data))

        fig = _base_layout(go.Figure(), "Allure par segment")
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

        fig = _base_layout(make_subplots(rows=2, cols=1), "Prediction vs True Target")
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
                hovertemplate="Distance : %{x:.2f} km<br>Predicted target : %{y:.1f} m/s<extra></extra>",
            ),
            row=1,
            col=1,
        )
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
            for zone in HearthRateZoneLabel.__args__
        }

        total = sum(durations.values())
        values = np.array(list(durations.values()))
        fig = _base_layout(go.Figure(), "Temps passé dans les zones cardiaques")
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


# ================================================================================= #
#                                   MINETTI PLOT
# ================================================================================= #
def plot_minetti_curve() -> go.Figure:
    """Plot the Minetti energy costs relative to flat terrain.

    Returns
    -------
    plotly.graph_objects.Figure
        Walking and running costs for slopes from -35 % to 35 %.

    See Also
    --------
    TrailPredictor.minetti.cost_of_running : Running energy cost.
    TrailPredictor.minetti.cost_of_walking : Minimum walking energy cost.
    """
    slopes = np.linspace(-0.35, 0.35, 100)
    walking_cost = cost_of_walking(slopes)
    running_cost = cost_of_running(slopes)

    fig = _base_layout(go.Figure(), "Minetti model")
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=walking_cost / cost_of_walking(0),
            mode="lines",
            name="Cost_walk",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=running_cost / cost_of_running(0),
            mode="lines",
            name="Cost_running",
        )
    )
    fig.update_xaxes(title="Slope")
    fig.update_yaxes(title="Energy cost relative to flat terrain")
    return fig
