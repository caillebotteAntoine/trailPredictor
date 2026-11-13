"""Vulture whitelist: code used outside the scanned paths, or false positives.

Vulture only scans ``TrailPredictor``, ``pages`` and ``app.py``, so the API
called from the notebooks (``main.ipynb``, ``prediction_from_gpx.ipynb``)
looks unused. Each line below marks a name as used; remove it when the
notebooks stop using it.
"""

# pylint: disable=pointless-statement

import plotly.graph_objects as go

from TrailPredictor.dataset import Dataset
from TrailPredictor.plot import PlotActivities, PlotDataset, plot_minetti_curve

# --- Used in the notebooks
Dataset.get_preprocess
Dataset.predictors
Dataset.true_total_time
Dataset.predict_total_time
Dataset.get_activities
PlotDataset.vs_prediction
PlotActivities.error_prediction
plot_minetti_curve

# --- False positives
# Attribute of a plotly trace, set in PlotDataset.merge_figure.
go.Scatter.showlegend
