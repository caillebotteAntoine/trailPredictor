from typing import Literal, TypeAlias

from collections.abc import Callable

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from TrailPredictor.activities import RunningActivities
from TrailPredictor.chrono import Speed
from TrailPredictor.dataset import Dataset, Target, TARGETS
from TrailPredictor.minetti import cost_of_running, cost_of_walking

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
    if not np.isfinite(pace):
        return "—"
    minutes, seconds = divmod(int(round(pace * 60)), 60)
    return f"{minutes}:{seconds:02d}/km"


def _pace_ticks(values: pd.Series) -> tuple[list[float], list[str]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return [], []
    low = np.floor(numeric.min() * 2) / 2
    high = np.ceil(numeric.max() * 2) / 2
    ticks = np.arange(low, high + 0.25, 0.5)
    return ticks.tolist(), [_pace_text(value) for value in ticks]


TAGETS_UNITS = {
    "pace": "min/km",
    "log_pace": "log(min/km)",
    "speed": "km/h",
    "log_speed": "log(km/h)",
}
TARGETS_COLORS = {
    "pace": "#56B4E9",
    "log_pace": "#E95656",
    "speed": "#5BE956",
    "log_speed": "#E456E9",
}


class PlotDataset(Dataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset._data, target=dataset._target)

    @property
    def target_label(self) -> str:
        return f"{self._target} ({TAGETS_UNITS[self._target]})"

    def merge_figure(
        self, plot_fct: Callable[[], go.Figure], title: str, **kwargs
    ) -> go.Figure:
        """Merge a figure with the target histogram."""
        merged = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=TARGETS,
            vertical_spacing=0.15,
        )
        current_target = self._target
        figs = {}
        try:
            for col in TARGETS:
                self.change_target(col)
                figs[col] = plot_fct(**kwargs)
        finally:
            self.change_target(current_target)

        for i, fig in enumerate(figs.values(), start=1):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
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

    def target_histogram(self) -> go.Figure:
        """Display a histogram of the target variable."""
        fig = _base_layout(go.Figure(), f"Histogram of {self._target}")
        fig.add_trace(
            go.Histogram(
                x=self.target,
                nbinsx=50,
                marker={"color": TARGETS_COLORS[self._target]},
                name=self.target_label,
            )
        )
        fig.update_xaxes(title=self._target)
        fig.update_yaxes(title="Count")
        return fig

    def target_vs_feat(self, features="altitude_drop") -> go.Figure:
        """Display a scatter plot of the target variable vs altitude drop."""
        if features not in self._data.columns:
            raise ValueError(f"Invalid feature: {features}")
        values = self._data[features]
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"Feature '{features}' must be numeric")
        if values.dropna().empty:
            raise ValueError(f"Feature '{features}' contains no numeric values")
        fig = _base_layout(go.Figure(), f"{self._target} vs {features} per segment")
        fig.add_trace(
            go.Scatter(
                x=values,
                y=self.target,
                mode="markers",
                marker={"color": TARGETS_COLORS[self._target], "size": 5},
                name=self.target_label,
            )
        )
        fig.update_xaxes(title=features)
        fig.update_yaxes(title=self.target_label)
        return fig


class PlotActivities(RunningActivities):
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
                "pace",
                "heart_rate",
            ]
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
    def iter_drop_categories(self):
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
        """Display the GPS track on an OpenStreetMap background."""
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
        """Display segment pace with bars sized by covered distance."""
        data = self._data.copy()
        distance = data["distance"]
        pace = pd.to_numeric(pd.Series([p.as_pace() for p in self._data["pace"]]))
        # set to NA the pace bigger than 20 min/km
        pace[pace > Speed.from_pace(20).as_pace()] = np.nan
        average = pace.mean()

        # durations = data["elapsed_time"].dt.total_seconds()
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
        """Overlay heart rate and altitude on two y axes."""

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
        """Overlay speed (true and predicted is available) and altitude on two y axes."""

        data = self._data.copy()
        x = self.distance_km
        fig = _base_layout(
            make_subplots(specs=[[{"secondary_y": True}]]),
            "Vitesse et altitude",
        )

        customdata = data[["pace"]]

        hovertemplate = (
            "Distance : %{x:.2f} km<br>"
            "Altitude : %{y:.1f} m<br>"
            "Speed : %{customdata:.1f} km/h<extra></extra>"
        )
        if self.prediction is not None:
            customdata["prediction"] = self.prediction
            customdata["error"] = np.abs(self.prediction - data["pace"])
            hovertemplate = hovertemplate.replace(
                "Speed : %{customdata:.1f} km/h<extra></extra>",
                "Speed : %{customdata[0]:.1f} km/h"
                "<br>Prediction : %{customdata[1]:.1f} km/h"
                "<br>Erreur : %{customdata[2]:.1f} km/h<extra></extra>",
            )

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
                y=data["pace"],
                mode="lines",
                name="Speed",
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
                    name="Speed predicted",
                    line={"color": "#8256E9"},
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                secondary_y=True,
            )

        fig.update_xaxes(title="Distance (km)")
        fig.update_yaxes(title="Speed (m/s)", secondary_y=False)
        fig.update_yaxes(title="Altitude (m)", secondary_y=True)
        return fig

    # ================================================================================== #
    #                                   TIME IN HEART RATE ZONES PLOT
    # ================================================================================== #
    def time_in_hr_zones(self) -> go.Figure:
        """Display time spent in heart-rate zones."""

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



def plot_minetti_curve() -> go.Figure:
    """Plot Minetti energy costs relative to flat terrain."""
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
