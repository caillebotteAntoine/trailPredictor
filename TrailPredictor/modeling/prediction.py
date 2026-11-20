"""Prediction of the target of a planned activity, e.g. a race route."""

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from TrailPredictor.activities import RunningActivities
from TrailPredictor.dataset import Dataset
from TrailPredictor.features import build_features
from TrailPredictor.modeling.lasso import LassoConfig, ModelingLasso
from TrailPredictor.utils import MODELS_DIR

logger = logging.getLogger(__name__)

# Features unknown before the start of a race: "pause" is derived from the
# speed, i.e. from the target. The heart-rate zone is unknown too, but the
# preprocessing always encodes it: a target zone is given instead.
UNKNOWN_BEFORE_START = ("pause",)


def predict_activity(
    dataset: Dataset,
    activity: RunningActivities,
    heart_rate_zone: str,
    config: LassoConfig | None = None,
    models_dir: Path = MODELS_DIR,
) -> RunningActivities:
    """Predict the target of each segment of a planned activity.

    A relaxed Lasso (Lasso selection then linear regression) is fitted on
    every activity of `dataset`, using only the features known before the
    start, then applied to the features of `activity`. As the heart rate of
    a planned activity is unknown, every segment is given the target
    `heart_rate_zone`.

    Parameters
    ----------
    dataset : Dataset
        Training activities; its target (e.g. ``"pace"``) is the predicted
        one. It is not modified.
    activity : RunningActivities
        Planned activity, e.g. from ``RunningActivities.from_gpx_file``, with
        at least the ``distance``, ``altitude``, ``drop_pos``, ``drop_neg``,
        ``speed`` and ``heart_rate`` columns (time columns may be missing
        values). It is not modified.
    heart_rate_zone : str
        Heart-rate zone targeted on the whole activity, one of the zones of
        `dataset` (e.g. ``"Aérobic (Endur.)"`` with the default zones).
    config : LassoConfig or None, optional
        Configuration of the Lasso. Default is None, meaning
        ``LassoConfig()``.
    models_dir : pathlib.Path, optional
        Root directory of the saved model results. Default is ``MODELS_DIR``.

    Returns
    -------
    RunningActivities
        A copy of `activity` whose prediction is named ``"<target>_pred"``,
        in the unit of the target (e.g. min/km for ``"pace"``).

    Raises
    ------
    ValueError
        If the segment widths of `dataset` and `activity` differ, or
        `heart_rate_zone` is not a zone of `dataset`.

    See Also
    --------
    TrailPredictor.modeling.lasso.ModelingLasso : Model fitted here.

    Notes
    -----
    The features of `activity` are built with the default parameters of
    ``build_features``; the dataset must have been built with the same ones.
    """
    if not np.isclose(dataset.segment_width, activity.segment_width):
        raise ValueError(
            f"Segment width of the activity ({activity.segment_width} m) differs"
            f" from the one of the dataset ({dataset.segment_width} m)."
        )

    # shallow copy: the features setter only replaces the list of names
    train = copy.copy(dataset)
    train.features = [f for f in dataset.features if f not in UNKNOWN_BEFORE_START]
    pipeline = ModelingLasso(train, config, models_dir=models_dir).fit()

    predictors = build_features(activity.data).assign(heart_rate_zone=heart_rate_zone)[
        train.features
    ]
    prediction = pd.Series(
        pipeline.predict(predictors),
        index=activity.data.index,
        name=f"{train.target_name}_pred",
        dtype="float64",
    )
    logger.info(
        "Predicted %s of %d segments of '%s'",
        train.target_name,
        len(prediction),
        activity.name,
    )
    return RunningActivities(activity.data, prediction=prediction)
