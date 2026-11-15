"""Plotly figure of the Minetti energy-cost curves."""

import numpy as np
import plotly.graph_objects as go

from TrailPredictor.minetti import cost_of_running, cost_of_walking
from TrailPredictor.plot._common import _base_layout


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
