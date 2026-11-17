"""Plotly figure of the Lasso regularisation path."""

from functools import cached_property

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from TrailPredictor.modeling.lasso import LassoPath, ModelingLasso
from TrailPredictor.plot._common import CATEGORICAL_COLORS, TARGETS_UNITS, _base_layout

OTHER_FEATURES_COLOR = "#9e9e9e"
CV_COLOR = "#1f4e79"
CV_BAND_COLOR = "rgba(31, 78, 121, 0.2)"  # CV_COLOR, transparent
ALPHA_MARKER_COLOR = "#333333"  # neutral: the palette colours identify features
ZERO_LINE_COLOR = "#4d4d4d"
# the line style changes once the palette is exhausted
LINE_DASHES = ("solid", "dash", "dashdot", "longdash")
HIGHLIGHT_WIDTH = 2
OTHER_WIDTH = 1
OTHER_OPACITY = 0.5
PATH_ROW_HEIGHT = 0.68  # share of the figure height for the coefficient path


class PlotLasso(ModelingLasso):
    """Plotting interface of the Lasso feature selection.

    Parameters
    ----------
    model : ModelingLasso
        Model to plot; its dataset and configuration are shared.

    See Also
    --------
    TrailPredictor.modeling.lasso.ModelingLasso : Lasso feature selection.
    """

    def __init__(self, model: ModelingLasso):
        super().__init__(model.dataset, model.config, model.models_dir)

    @cached_property
    def path(self) -> LassoPath:
        """Return the regularisation path, fitted once on the whole dataset."""
        return self.fit_regularization_path()

    # ================================================================================= #
    #                               REGULARISATION PATH
    # ================================================================================= #
    @staticmethod
    def _add_alpha_marker(  # pylint: disable=too-many-arguments
        fig: go.Figure,
        alpha: float,
        y_spans: tuple[tuple[float, float], tuple[float, float]],
        name: str,
        *,
        dash: str,
    ) -> None:
        """Draw a vertical line at `alpha` on both rows, with one legend entry.

        Drawn as traces rather than shapes so it gets a legend entry and
        stays at its data position on the log axis.
        """
        for row, (y_low, y_high) in enumerate(y_spans, start=1):
            fig.add_trace(
                go.Scatter(
                    x=[alpha, alpha],
                    y=[y_low, y_high],
                    mode="lines",
                    line={"color": ALPHA_MARKER_COLOR, "dash": dash, "width": 1.5},
                    name=name,
                    legend="legend2",
                    legendgroup=name,
                    showlegend=row == 1,
                    hovertemplate=f"{name}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    def _add_coefficient_paths(
        self, fig: go.Figure, highlighted: list[str], n_active: npt.NDArray[np.int64]
    ) -> None:
        """Draw the highlighted features in colour, the others in grey."""
        coef_path = self.path.coef_path
        alphas = self.path.alphas
        others = coef_path.columns.difference(highlighted)
        # loop over the features (one trace each), not over the data rows
        for rank, feature in enumerate(others):
            fig.add_trace(
                go.Scatter(
                    x=alphas,
                    y=coef_path[feature],
                    mode="lines",
                    line={"color": OTHER_FEATURES_COLOR, "width": OTHER_WIDTH},
                    opacity=OTHER_OPACITY,
                    name="Autres variables",
                    legendgroup="other_features",
                    showlegend=rank == 0,
                    customdata=n_active,
                    hovertemplate=(
                        f"{feature}<br>α : %{{x:.3g}}<br>coefficient : %{{y:.3f}}"
                        "<br>%{customdata} variables actives<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        n_colors = len(CATEGORICAL_COLORS)
        for rank, feature in enumerate(highlighted):
            fig.add_trace(
                go.Scatter(
                    x=alphas,
                    y=coef_path[feature],
                    mode="lines",
                    line={
                        "color": CATEGORICAL_COLORS[rank % n_colors],
                        "dash": LINE_DASHES[(rank // n_colors) % len(LINE_DASHES)],
                        "width": HIGHLIGHT_WIDTH,
                    },
                    name=feature,
                    customdata=n_active,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>α : %{x:.3g}"
                        "<br>coefficient : %{y:.3f}"
                        "<br>%{customdata} variables actives<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        fig.add_hline(y=0, line={"color": ZERO_LINE_COLOR, "width": 1}, row=1, col=1)

    def _add_cv_error(self, fig: go.Figure) -> None:
        """Draw the mean CV error, its ±1 SE band and the 1-SE threshold."""
        path = self.path
        mse_mean, mse_se = path.mse_mean, path.mse_se
        fig.add_trace(
            go.Scatter(
                x=path.alphas,
                y=mse_mean + mse_se,
                mode="lines",
                line={"width": 0},
                legend="legend2",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=path.alphas,
                y=mse_mean - mse_se,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=CV_BAND_COLOR,
                name="± 1 erreur-type",
                legend="legend2",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=path.alphas,
                y=mse_mean,
                mode="lines",
                line={"color": CV_COLOR, "width": HIGHLIGHT_WIDTH},
                name="Erreur CV moyenne",
                legend="legend2",
                customdata=mse_se,
                hovertemplate=(
                    "α : %{x:.3g}<br>MSE CV : %{y:.4g} ± %{customdata:.2g}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[path.alpha, path.alpha_1se],
                y=[path.mse_threshold, path.mse_threshold],
                mode="lines",
                line={"color": ALPHA_MARKER_COLOR, "dash": "dot", "width": 1},
                name="MSE min + 1 erreur-type",
                legend="legend2",
                hovertemplate="MSE min + 1 erreur-type : %{y:.4g}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    def regularization_path(self, n_highlight: int | None = None) -> go.Figure:
        """Plot the coefficient path and the cross-validation error along α.

        Parameters
        ----------
        n_highlight : int or None, optional
            Number of features drawn in colour, the largest in absolute
            coefficient at the 1-SE α; the others are grey. Default is None,
            meaning every feature active at the 1-SE α.

        Returns
        -------
        plotly.graph_objects.Figure
            Two rows sharing the log α axis: the standardised coefficient of
            each feature (top), and the mean inner CV error with its ±1
            standard error band (bottom). The α selected by ``LassoCV`` and
            the α of the "one standard error" rule are marked on both rows.

        Raises
        ------
        ValueError
            If `n_highlight` is negative.

        Notes
        -----
        The path is fitted on the whole dataset, not on the outer folds of
        ``evaluate_selection``: the first call fits ``LassoCV``, later calls
        reuse it.
        """
        if n_highlight is not None and n_highlight < 0:
            raise ValueError(f"n_highlight must be non-negative, got {n_highlight}")
        path = self.path
        coefs_1se = path.coefs_at(path.alpha_1se).abs()
        if n_highlight is None:
            n_highlight = int((coefs_1se > path.threshold).sum())
        highlighted = coefs_1se.nlargest(n_highlight).index.tolist()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[PATH_ROW_HEIGHT, 1 - PATH_ROW_HEIGHT],
        )
        _base_layout(
            fig,
            "Chemin de régularisation du Lasso",
            "Coefficients standardisés ; α choisi par validation croisée "
            "groupée par activité",
        )
        fig.update_layout(
            legend={
                "title": {"text": "Variables"},
                "orientation": "v",
                "x": 1.02,
                "y": 1,
                "yanchor": "top",
            },
            legend2={"orientation": "v", "x": 1.02, "y": 0.3, "yanchor": "top"},
            margin={"r": 250},
        )

        self._add_coefficient_paths(fig, highlighted, path.n_active.to_numpy())
        self._add_cv_error(fig)

        coef_values = path.coef_path.to_numpy()
        y_spans = (
            (float(coef_values.min()), float(coef_values.max())),
            (
                float((path.mse_mean - path.mse_se).min()),
                float((path.mse_mean + path.mse_se).max()),
            ),
        )
        self._add_alpha_marker(
            fig, path.alpha, y_spans, f"α retenu (CV) = {path.alpha:.3g}", dash="dash"
        )
        self._add_alpha_marker(
            fig,
            path.alpha_1se,
            y_spans,
            f"α 1 erreur-type = {path.alpha_1se:.3g}",
            dash="dot",
        )

        unit = TARGETS_UNITS[self.dataset.target_name]
        fig.update_xaxes(type="log")
        fig.update_xaxes(title="α (échelle log)", row=2, col=1)
        fig.update_yaxes(title="Coefficient standardisé", row=1, col=1)
        fig.update_yaxes(
            title=f"MSE CV {self.dataset.target_name} [({unit})²]", row=2, col=1
        )
        return fig
