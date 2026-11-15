"""Plotly figures for the dataset, the activities and the Minetti curves."""

from TrailPredictor.plot.activities import PlotActivities
from TrailPredictor.plot.dataset import PlotDataset
from TrailPredictor.plot.minetti import plot_minetti_curve

__all__ = ["PlotActivities", "PlotDataset", "plot_minetti_curve"]
