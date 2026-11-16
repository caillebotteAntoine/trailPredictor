"""Vulture whitelist: code used outside the scanned paths, or false positives.

Vulture only scans ``TrailPredictor``, ``pages`` and ``app.py``, so the API
called from the notebooks (``main.ipynb``, ``prediction_from_gpx.ipynb``)
looks unused. Each line below marks a name as used; remove it when the
notebooks stop using it.
"""

# pylint: disable=pointless-statement,protected-access

import plotly.graph_objects as go

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling import candidates
from TrailPredictor.modeling.config import ModelConfig
from TrailPredictor.modeling.heart_rate import HeartRateConfig, ModelingHeartRate
from TrailPredictor.plot import (
    PlotActivities,
    PlotDataset,
    PlotHeartRate,
    plot_minetti_curve,
)

# --- Used in the notebooks
Dataset.get_preprocess
Dataset.predictors
Dataset.true_total_time
Dataset.predict_total_time
Dataset.get_activities
Dataset.modeling
PlotDataset.vs_prediction
PlotActivities.error_prediction
plot_minetti_curve
ModelingHeartRate.optimize_zones

# --- Future use in the notebooks
candidates.make_models
PlotActivities.iter_heart_rate_zones
PlotHeartRate.from_model
PlotHeartRate.trials_zones
PlotHeartRate.best_zones_per_n

# --- False positives
# Attribute of a plotly trace, set in PlotDataset.merge_figure.
go.Scatter.showlegend
# Read by pydantic: model settings and validator registered by its decorator.
ModelConfig.model_config
HeartRateConfig._check_first_cut_below_max
