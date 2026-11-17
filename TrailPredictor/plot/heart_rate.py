"""Plotly figures of the heart-rate zone optimisation."""

import json
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from TrailPredictor.modeling.heart_rate import (
    HR_CEIL_BPM,
    HR_FLOOR_BPM,
    REFERENCE_CUTS,
    ModelingHeartRate,
)
from TrailPredictor.plot._common import CATEGORICAL_COLORS, _base_layout

# colours of the zones Z0 ... Z9, shared by every figure
ZONE_COLORS = CATEGORICAL_COLORS
BEST_COLOR = "green"
BEST_RMSE_COLOR = "#b41f1f"
REFERENCE_COLOR = "gray"
BAR_WIDTH = 0.8
Y_MARGIN_BPM = 5  # margin around the cuts on the zone axis
TRIAL_COLUMNS = ["n_zones", "trial_number", "rmse", "rmse_se", "cuts", "state"]


def _parse_cuts(cuts: object) -> list[int] | None:
    """Return the cuts as a list, decoding the string stored in the CSV."""
    if isinstance(cuts, str):
        return [int(cut) for cut in json.loads(cuts)]
    if isinstance(cuts, Sequence | np.ndarray):
        return [int(cut) for cut in cuts]
    return None


def _zone_bounds(cuts: pd.Series) -> npt.NDArray[np.float64]:
    """Stack the zone bounds ``[floor, *cuts, ceil]`` of each row.

    Rows with fewer zones are padded with NaN, so zone ``k`` spans
    ``bounds[:, k]`` to ``bounds[:, k + 1]`` and is missing where undefined.
    """
    n_bounds = int(cuts.map(len).max()) + 2
    # one short list per trial: cuts are ragged lists, no vectorized constructor
    bounds = np.array(
        [
            [HR_FLOOR_BPM, *row_cuts, HR_CEIL_BPM]
            + [np.nan] * (n_bounds - len(row_cuts) - 2)
            for row_cuts in cuts
        ],
        dtype=np.float64,
    )
    assert bounds.shape == (len(cuts), n_bounds)
    return bounds


class PlotHeartRate:
    """Plotting interface of the heart-rate zone optimisation.

    Parameters
    ----------
    trials : pandas.DataFrame
        One row per Optuna trial, as ``ModelingHeartRate.result``: columns
        ``n_zones``, ``trial_number``, ``rmse``, ``rmse_se``, ``cuts`` (list of
        cuts in bpm, or its JSON string when read from CSV) and ``state``.
    target_name : str
        Name of the target the RMSE is computed on.
    reference_rmse : float or None, optional
        RMSE of the reference zones (``REFERENCE_CUTS``), drawn as a dashed
        line when given. Default is None.

    Raises
    ------
    ValueError
        If a column of `trials` is missing.

    See Also
    --------
    TrailPredictor.modeling.heart_rate.ModelingHeartRate : Zone optimisation.
    """

    def __init__(
        self,
        trials: pd.DataFrame,
        target_name: str,
        reference_rmse: float | None = None,
    ):
        missing = set(TRIAL_COLUMNS) - set(trials.columns)
        if missing:
            raise ValueError(f"Missing trial columns: {sorted(missing)}")
        self._trials = trials.reset_index(drop=True).assign(
            cuts=lambda df: df["cuts"].map(_parse_cuts)
        )
        self._target_name = target_name
        self._reference_rmse = reference_rmse

    @classmethod
    def from_model(
        cls, model: ModelingHeartRate, reference_rmse: float | None = None
    ) -> "PlotHeartRate":
        """Build the plotting interface from a fitted model.

        Parameters
        ----------
        model : ModelingHeartRate
            Model whose ``result`` holds the trials.
        reference_rmse : float or None, optional
            RMSE of the reference zones. Default is None.

        Returns
        -------
        PlotHeartRate
            Plotting interface of the model trials.

        Raises
        ------
        ValueError
            If the model has no result yet.
        """
        if model.result is None:
            raise ValueError("The model has no result: run optimize_zones first.")
        return cls(
            model.result, model.dataset.target_name, reference_rmse=reference_rmse
        )

    @property
    def completed_trials(self) -> pd.DataFrame:
        """Return the completed trials, sorted by zone count and trial number."""
        return self._trials.query("state == 'COMPLETE'").sort_values(
            ["n_zones", "trial_number"]
        )

    @property
    def best_trials(self) -> pd.DataFrame:
        """Return the completed trial with the lowest RMSE for each zone count.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``n_zones``, sorted, with the trial columns.
        """
        completed = self.completed_trials
        return (
            completed.loc[completed.groupby("n_zones")["rmse"].idxmin()]
            .set_index("n_zones")
            .sort_index()
        )

    # ================================================================================= #
    #                                  SHARED HELPERS
    # ================================================================================= #
    def _zones_and_rmse_figure(self, title: str) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        _base_layout(fig, title)
        fig.update_layout(
            barmode="overlay",
            legend={
                "title": {"text": "Zones cardiaques"},
                "orientation": "v",
                "x": 1.02,
                "y": 1,
                "yanchor": "top",
            },
            legend2={"orientation": "v", "x": 1.02, "y": 0.45, "yanchor": "top"},
            margin={"r": 200},
        )
        fig.update_yaxes(title="Seuils cardiaques (bpm)", row=1, col=1)
        fig.update_yaxes(title=f"RMSE ({self._target_name})", row=2, col=1)
        return fig

    @staticmethod
    def _add_zone_bars(
        fig: go.Figure,
        positions: npt.NDArray[np.int64],
        cuts: pd.Series,
        hover_labels: Sequence[str],
    ) -> None:
        """Draw one floating bar per position and zone on the first row."""
        bounds = _zone_bounds(cuts)
        # loop over the zone index (parameters), not over the data rows
        for k in range(bounds.shape[1] - 1):
            low, high = bounds[:, k], bounds[:, k + 1]
            fig.add_trace(
                go.Bar(
                    x=positions,
                    y=high - low,
                    base=low,
                    width=BAR_WIDTH,
                    marker={"color": ZONE_COLORS[k % len(ZONE_COLORS)]},
                    name=f"Z{k}",
                    customdata=np.column_stack([low, high]),
                    hovertext=hover_labels,
                    hovertemplate=(
                        f"%{{hovertext}}<br>Z{k} : %{{customdata[0]:.0f}}"
                        "–%{customdata[1]:.0f} bpm<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        all_cuts = bounds[:, 1:-1]
        fig.update_yaxes(
            range=[
                np.nanmin(all_cuts) - Y_MARGIN_BPM,
                np.nanmax(all_cuts) + Y_MARGIN_BPM,
            ],
            row=1,
            col=1,
        )

    @staticmethod
    def _add_reference_cuts(fig: go.Figure, n_positions: int) -> None:
        """Draw the reference cuts, coloured like the zone they open."""
        x_span = [-0.5, n_positions - 0.5]
        # cut c_{k+1} is the lower bound of Z{k+1}; single legend entry
        for k, cut in enumerate(REFERENCE_CUTS):
            fig.add_trace(
                go.Scatter(
                    x=x_span,
                    y=[cut, cut],
                    mode="lines",
                    line={
                        "color": ZONE_COLORS[(k + 1) % len(ZONE_COLORS)],
                        "dash": "dot",
                        "width": 1,
                    },
                    name="Seuils référence",
                    legendgroup="reference_cuts",
                    showlegend=k == 0,
                    hovertemplate=f"Seuil référence : {cut} bpm<extra></extra>",
                ),
                row=1,
                col=1,
            )

    def _add_reference_rmse(self, fig: go.Figure, n_positions: int) -> None:
        if self._reference_rmse is None:
            return
        fig.add_trace(
            go.Scatter(
                x=[-0.5, n_positions - 0.5],
                y=[self._reference_rmse, self._reference_rmse],
                mode="lines",
                line={"color": REFERENCE_COLOR, "dash": "dash"},
                name=f"RMSE référence (N={len(REFERENCE_CUTS) + 1})",
                legend="legend2",
                hovertemplate="RMSE référence : %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    @staticmethod
    def _add_best_marker(fig: go.Figure, position: int, label: str) -> None:
        """Highlight the best position with a vertical line across both rows."""
        fig.add_vline(
            x=position,
            line={"color": BEST_COLOR, "dash": "dash", "width": 1},
            row="all",
            col=1,
        )
        fig.add_annotation(
            x=position,
            y=1,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            yanchor="bottom",
            font={"color": BEST_COLOR},
        )

    # ================================================================================= #
    #                               TRIALS ZONES PLOT
    # ================================================================================= #
    def trials_zones(self, n_zones: int) -> go.Figure:
        """Plot the cuts and the RMSE of every completed trial for one zone count.

        Parameters
        ----------
        n_zones : int
            Number of heart-rate zones of the trials to plot.

        Returns
        -------
        plotly.graph_objects.Figure
            Two rows sharing the trial axis: the zones of each trial as
            floating stacked bars with the reference cuts, and the RMSE of
            each trial with its running minimum. The best trial is marked.

        Raises
        ------
        ValueError
            If no trial with `n_zones` zones is completed.
        """
        trials = self.completed_trials.query("n_zones == @n_zones")
        if trials.empty:
            raise ValueError(f"No completed trial with {n_zones} zones")
        trial_numbers = trials["trial_number"].to_numpy()
        rmse = trials["rmse"].to_numpy(dtype=np.float64)
        # evenly spaced positions: pruned trials leave gaps in trial_number
        positions = np.arange(len(trials))

        fig = self._zones_and_rmse_figure(f"RMSE et seuils cardiaques pour N={n_zones}")
        self._add_zone_bars(
            fig, positions, trials["cuts"], [f"Essai {n}" for n in trial_numbers]
        )
        self._add_reference_cuts(fig, len(positions))

        fig.add_trace(
            go.Scatter(
                x=positions,
                y=rmse,
                mode="markers",
                name=f"RMSE (N={n_zones})",
                legend="legend2",
                customdata=trial_numbers,
                hovertemplate="Essai %{customdata}<br>RMSE : %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=np.minimum.accumulate(rmse),
                mode="lines+markers",
                marker={"size": 4},
                line={"color": BEST_RMSE_COLOR},
                name=f"RMSE min (N={n_zones})",
                legend="legend2",
                hovertemplate="RMSE min : %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        self._add_reference_rmse(fig, len(positions))

        best_position = int(np.argmin(rmse))
        self._add_best_marker(
            fig, best_position, f"RMSE min (essai {trial_numbers[best_position]})"
        )
        fig.update_xaxes(
            tickvals=positions,
            ticktext=trial_numbers.astype(str),
            tickangle=-90,
            tickfont={"size": 8},
            showticklabels=True,
        )
        fig.update_xaxes(title="Numéro d'essai", row=2, col=1)
        return fig

    # ================================================================================= #
    #                              BEST ZONES PER N PLOT
    # ================================================================================= #
    def best_zones_per_n(self) -> go.Figure:
        """Plot the cuts and the RMSE of the best trial for each zone count.

        Returns
        -------
        plotly.graph_objects.Figure
            Two rows sharing the zone-count axis: the zones of the best trial
            as floating stacked bars with the reference cuts, and its RMSE
            with the standard error over the folds. The best N is marked.

        Raises
        ------
        ValueError
            If no trial is completed.
        """
        best = self.best_trials
        if best.empty:
            raise ValueError("No completed trial")
        n_values = best.index.to_numpy()
        rmse = best["rmse"].to_numpy(dtype=np.float64)
        positions = np.arange(len(best))

        fig = self._zones_and_rmse_figure(
            "RMSE et seuils cardiaques du meilleur essai pour chaque N"
        )
        self._add_zone_bars(fig, positions, best["cuts"], [f"N={n}" for n in n_values])
        self._add_reference_cuts(fig, len(positions))

        fig.add_trace(
            go.Scatter(
                x=positions,
                y=rmse,
                error_y={"type": "data", "array": best["rmse_se"].to_numpy()},
                mode="lines+markers",
                name="RMSE meilleur essai ± SE",
                legend="legend2",
                customdata=np.column_stack(
                    [n_values, best["trial_number"], best["rmse_se"]]
                ),
                hovertemplate=(
                    "N=%{customdata[0]:.0f} (essai %{customdata[1]:.0f})<br>"
                    "RMSE : %{y:.4f} ± %{customdata[2]:.4f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        self._add_reference_rmse(fig, len(positions))

        best_position = int(np.argmin(rmse))
        self._add_best_marker(
            fig, best_position, f"RMSE min (N={n_values[best_position]})"
        )
        fig.update_xaxes(
            tickvals=positions, ticktext=n_values.astype(str), showticklabels=True
        )
        fig.update_xaxes(title="Nombre de zones N", row=2, col=1)
        return fig
