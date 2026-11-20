"""Vulture whitelist: code used outside the scanned paths, or false positives.

Vulture only scans ``TrailPredictor``, ``pages``, ``app.py`` and
``desktop.py``, so the API called from the notebooks (``notebooks/*.ipynb``)
looks unused. Each line below marks a name as used; remove it when the
notebooks stop using it.
"""

# pylint: disable=pointless-statement,protected-access

import plotly.graph_objects as go
import uvicorn

from TrailPredictor.api import activity_routes
from TrailPredictor.api import app as api_app
from TrailPredictor.api.activity_service import ActivitySummary, TrackView
from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling import candidates
from TrailPredictor.modeling.config import ModelConfig
from TrailPredictor.modeling.heart_rate import HeartRateConfig, ModelingHeartRate
from TrailPredictor.modeling.lasso import ModelingLasso
from TrailPredictor.plot import (
    PlotActivities,
    PlotDataset,
    PlotHeartRate,
    PlotLasso,
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
ModelingLasso.evaluate_selection
ModelingLasso.plot
PlotLasso.regularization_path

# --- False positives
# Attribute of a plotly trace, set in PlotDataset.merge_figure.
go.Scatter.showlegend
# Read by pydantic: model settings and validator registered by its decorator.
ModelConfig.model_config
HeartRateConfig._check_first_cut_below_max
# FastAPI routes, registered by their decorator.
api_app.health
api_app.job_status
api_app.job_event_stream
api_app.package_script
activity_routes.list_activities
activity_routes.activity_charts
activity_routes.activity_track
# Fields of API responses, serialized to JSON by FastAPI.
ActivitySummary.id
TrackView.tooltip
# Read by the uvicorn server loop, set in BackgroundServer.stop.
uvicorn.Server.should_exit  # pylint: disable=no-member  # instance attribute
