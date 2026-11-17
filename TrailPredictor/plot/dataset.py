"""Plotly figures for a whole dataset."""

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from TrailPredictor.dataset import Dataset
from TrailPredictor.plot._common import TARGETS_COLORS, TARGETS_UNITS, _base_layout
from TrailPredictor.speed_columns import TARGETS


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
            for target in TARGETS:
                self.target = target
                figs[target] = plot_fct(**kwargs)
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
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.2,
                "xanchor": "center",
                "x": 0.5,
            }
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
                hovertemplate="Prediction : %{x:.2f}<br>True value : %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # add unity line
        fig.add_trace(
            go.Scatter(
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
                hovertemplate="Prediction : %{x:.2f}<br>Residual : %{y:.2f}<extra></extra>",
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
