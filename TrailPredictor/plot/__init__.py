"""Plotly figures for the dataset, the activities, the heart-rate zones and Minetti."""

from TrailPredictor.plot.activities import PlotActivities
from TrailPredictor.plot.dataset import PlotDataset
from TrailPredictor.plot.heart_rate import PlotHeartRate
from TrailPredictor.plot.minetti import plot_minetti_curve

__all__ = ["PlotActivities", "PlotDataset", "PlotHeartRate", "plot_minetti_curve"]
