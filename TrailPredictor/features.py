"""Construction of the raw dataset and of the segment features."""

from typing import Literal

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
from sklearn.model_selection import train_test_split

from TrailPredictor.activities import RunningActivities
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.loading import load_activities
from TrailPredictor.minetti import cost_of_running, cost_of_walking

WINDOWS_KM = (1, 3, 5, 10)
FEATURES_SUFFIXES = ("done", "remaining")


HearthRateZone = Literal[
    "Unknown",
    "Récupération",
    "Aérobic (Endur.)",
    "Aérobic (Puiss.)",
    "Seuil",
    "Endurance anaérobie",
    "Anaérobie (Puiss.)",
]
HEART_RATE_ZONES: dict[HearthRateZone, tuple[int, int]] = {
    "Récupération": (0, 142),
    "Aérobic (Endur.)": (142, 161),
    "Aérobic (Puiss.)": (161, 170),
    "Seuil": (170, 183),
    "Endurance anaérobie": (183, 189),
    "Anaérobie (Puiss.)": (189, 210),
}
OUT_OF_ZONES_LABEL = "Unknown"  # zone of a missing heart rate, or outside every zone


def export_activities(
    activities=None,
    min_distance=30,
    max_distance=100,
    segment_width=500,
    progress_cb: ProgressCallback | None = None,
) -> list[RunningActivities]:
    """Load the running activities within a distance range.

    Parameters
    ----------
    activities : pandas.DataFrame, optional
        Activity summary as returned by ``load_activities``, loaded from the
        default CSV when omitted.
    min_distance : float, optional
        Minimum distance in km (exclusive). Default is 30.
    max_distance : float, optional
        Maximum distance in km (exclusive). Default is 100.
    segment_width : int, optional
        Segment length in metres. Default is 500.
    progress_cb : ProgressCallback, optional
        Progress reporter, also used to display the current activity.

    Returns
    -------
    list of RunningActivities
        One segmented activity per FIT file, longest first; activities
        without a FIT file are skipped.

    Raises
    ------
    AssertionError
        If `min_distance` is not less than `max_distance`.
    """
    if activities is None:
        activities = load_activities()

    assert min_distance < max_distance, "min_distance must be less than max_distance"

    act = activities[activities["type"] == "run"].sort_values(
        "distance_km", ascending=False
    )
    act = act[act["distance_km"] < max_distance]
    act = act[act["distance_km"] > min_distance]

    running_activities = []
    for position, (_, row) in enumerate(act.iterrows()):
        if progress_cb is not None:
            # report the current activity being processed
            progress_cb(
                position / len(act), f"{row['distance_km']:.2f} km - {row['name']}"
            )
        else:
            print(f"{row['distance_km']:.2f} km - {row['name']}")

        file_path = row["file_name"]

        if not file_path.endswith(".fit.gz") and not file_path.endswith(".fit"):
            print(
                f"⏩Skipping {file_path} ({row['name']}) because it is not a FIT file"
            )
            continue

        run = RunningActivities.from_computed_file(
            file_path, segment_width, progress_cb=progress_cb
        )
        run.name = row["name"]
        running_activities.append(run)

    return running_activities


def create_dataset(
    activities=None,
    min_distance=30,
    max_distance=100,
    segment_width=500,
    progress_cb: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Stack the segments of the running activities into one DataFrame.

    Parameters
    ----------
    activities : pandas.DataFrame, optional
        Activity summary, loaded from the default CSV when omitted.
    min_distance : float, optional
        Minimum distance in km (exclusive). Default is 30.
    max_distance : float, optional
        Maximum distance in km (exclusive). Default is 100.
    segment_width : int, optional
        Segment length in metres. Default is 500.
    progress_cb : ProgressCallback, optional
        Progress reporter.

    Returns
    -------
    pandas.DataFrame
        Segments of all activities, with an ``activity_id`` column.

    See Also
    --------
    export_activities : Loads the individual activities.
    """
    running_activities = export_activities(
        activities, min_distance, max_distance, segment_width, progress_cb
    )

    data = pd.concat(
        [
            activity.data.assign(activity_id=i)
            for i, activity in enumerate(running_activities)
        ],
        ignore_index=True,
    )
    return data


def split_test_train(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into a training and a test set by activity.

    All the segments of an activity end up in the same set.

    Parameters
    ----------
    data : pandas.DataFrame
        Segments with an ``activity_id`` column.
    test_size : float, optional
        Fraction of the activities in the test set. Default is 0.2.
    random_state : int, optional
        Seed of the split. Default is 42.

    Returns
    -------
    train_data : pandas.DataFrame
        Segments of the training activities.
    test_data : pandas.DataFrame
        Segments of the test activities.
    """
    activity_ids = data["activity_id"].unique()
    train_ids, test_ids = train_test_split(
        activity_ids, test_size=test_size, random_state=random_state
    )
    train_data = data.loc[data["activity_id"].isin(train_ids)]
    test_data = data.loc[data["activity_id"].isin(test_ids)]
    return train_data, test_data


def _window_sum(
    s: pd.Series | SeriesGroupBy, k: int, direction: str = "front"
) -> pd.Series:
    """Sum over the `k` previous ('back') or next ('front') segments.

    The current segment is excluded; at the edges, the available segments
    are summed.
    """
    if direction == "back":
        return s.transform(
            lambda values: values.shift(1).rolling(k, min_periods=1).sum().fillna(0.0)
        )
    return s.transform(
        lambda values: (
            values.iloc[::-1]
            .shift(1)
            .rolling(k, min_periods=1)
            .sum()
            .iloc[::-1]
            .fillna(0.0)
        )
    )


def assign_heart_rate_zones(
    heart_rate: pd.Series,
    heart_rate_zones: dict[str, tuple[int, int]]
    | dict[HearthRateZone, tuple[int, int]],
) -> pd.Series:
    """Label each heart rate with its zone.

    It only depends on the heart rate, so the zones can be changed without
    rebuilding the other features.

    Parameters
    ----------
    heart_rate : pandas.Series
        Heart rate of each segment, in bpm.
    heart_rate_zones : dict of str to tuple of (int, int)
        Heart-rate zones, mapping each zone label to its ``[low, high)``
        bounds in bpm.

    Returns
    -------
    pandas.Series
        Zone label of each segment, with the index of `heart_rate`;
        ``OUT_OF_ZONES_LABEL`` when the heart rate is missing or outside every
        zone.

    See Also
    --------
    build_features : Builds every predictor, including this one.
    """
    labels = np.select(
        [
            heart_rate.between(low, high, inclusive="left")
            for low, high in heart_rate_zones.values()
        ],
        list(heart_rate_zones.keys()),
        default=OUT_OF_ZONES_LABEL,
    )
    return pd.Series(labels, index=heart_rate.index, name="heart_rate_zone")


def build_features(
    dataset: pd.DataFrame,
    windows_km: tuple = WINDOWS_KM,
    drop_threshold: float = 2.0,
    heart_rate_zones: dict[str, tuple[int, int]]
    | dict[HearthRateZone, tuple[int, int]]
    | None = None,
) -> pd.DataFrame:
    """Build the predictors of each segment.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Segments with ``activity_id``, cumulative ``distance`` (metres) and
        ``altitude`` (metres) columns. It is not modified.
    windows_km : tuple of float, optional
        Window lengths in km of the back / front elevation sums. Default is
        ``WINDOWS_KM``.
    drop_threshold : float, optional
        Threshold for classifying a segment as uphill or downhill. Default is 2.0.
    heart_rate_zones : dict of str to tuple of (int, int), optional
        Heart-rate zones, mapping each zone label to its ``(low, high)``
        bounds in bpm. Default is None, meaning ``HEART_RATE_ZONES``.

    Returns
    -------
    pandas.DataFrame
        A copy of `dataset` with the elevation changes (per segment, done,
        remaining, total and their ratios), the distance remaining and total,
        the back / front window sums, the ``slope`` and the Minetti
        ``walking_cost`` / ``running_cost`` relative to flat terrain.

    See Also
    --------
    TrailPredictor.minetti.cost_of_running : Running energy cost.
    TrailPredictor.minetti.cost_of_walking : Walking energy cost.
    """
    if heart_rate_zones is None:
        heart_rate_zones = HEART_RATE_ZONES
    x = dataset.copy()

    if "activity_id" not in x.columns:
        x["activity_id"] = 0  # single activity

    x_grp = x.groupby("activity_id")

    segment_width = x["distance"].diff().median()  # segment width in meters

    # => difference in altitude between two consecutive points
    x["altitude_drop"] = x_grp["altitude"].diff().fillna(0)
    x["altitude_drop_pos"] = x["altitude_drop"].clip(lower=0)
    x["altitude_drop_neg"] = x["altitude_drop"].clip(upper=0)
    # cumulative of each variable for "done" and "remaining" features
    x_grp = x.groupby("activity_id")
    col_features = ("altitude_drop_pos", "altitude_drop_neg")
    for feature in col_features:
        x[f"{feature}_done"] = x_grp[feature].cumsum()
        x[f"{feature}_total"] = x_grp[feature].transform("sum")
        x[f"{feature}_remaining"] = x[f"{feature}_total"] - x[f"{feature}_done"]

    # special case for distance, which is already cumulative
    x["distance_done"] = x["distance"]
    x["distance_remaining"] = x_grp["distance"].transform(
        lambda values: values.max() - values
    )
    x = x.merge(
        x.groupby("activity_id")["distance"]
        .max()
        .reset_index()
        .rename(columns={"distance": "distance_total"})
    )

    # => ratio of each variable for "done" and "remaining" features
    # in order to remove the effect of the total value of each variable for each activity
    for feature in col_features:
        for suffix in FEATURES_SUFFIXES:
            col = f"{feature}_{suffix}"
            x[f"{col}_ratio"] = x[col] / x[f"{feature}_total"]

    # => sum of each variable for the previous and next k segments
    x_grp = x.groupby("activity_id")
    for feature in col_features:
        for w in windows_km:
            k = int(round(w * 1000 / segment_width))
            x[f"{feature}_back_{w}km"] = _window_sum(
                x_grp[feature], k, direction="back"
            )
            x[f"{feature}_front_{w}km"] = _window_sum(
                x_grp[feature], k, direction="front"
            )

    # => Minetti energy cost for running and walking
    x["slope"] = x["altitude_drop"] / segment_width
    x["walking_cost"] = cost_of_walking(x["slope"]) / cost_of_walking(0)
    x["running_cost"] = cost_of_running(x["slope"]) / cost_of_running(0)

    # if the speed is small and the cost of running is low, then it is likely a pause.
    # This can be used to filter out segments that are not representative of the activity.
    x["pause"] = (x["speed"] < 1.0) & (x["running_cost"] < 1.5)

    # removing done because it is redundant with distance
    x.drop(columns="distance_done", inplace=True)

    x["drop_category"] = np.select(
        [
            x["altitude_drop"] > drop_threshold,
            x["altitude_drop"] < -drop_threshold,
        ],
        ["Uphill", "Downhill"],
        default="Flat",
    )

    x["heart_rate_zone"] = assign_heart_rate_zones(x["heart_rate"], heart_rate_zones)

    if x["activity_id"].nunique() == 1:
        x.drop(columns="activity_id", inplace=True)

    return x
