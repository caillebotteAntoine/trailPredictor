import pandas as pd
from sklearn.model_selection import train_test_split


from TrailPredictor.activities import RunningActivities
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.loading import load_activities
from TrailPredictor.minetti import cost_of_walking, cost_of_running

WINDOWS_KM = (1, 3, 5, 10)


def export_activities(
    activities=None,
    min_distance=30,
    max_distance=100,
    segment_width=500,
    progress_cb: ProgressCallback | None = None,
) -> list[RunningActivities]:
    """Export running activities from a list of activities or load them from the default source."""
    if activities is None:
        activities = load_activities()

    assert min_distance < max_distance, "min_distance must be less than max_distance"

    act = activities[activities["type"] == "run"].sort_values(
        "distance_km", ascending=False
    )
    act = act[act["distance_km"] < max_distance]
    act = act[act["distance_km"] > min_distance]

    running_activities = []
    for _, row in act.iterrows():
        if progress_cb is not None:
            # change label of the progress bar to show the current activity being processed
            progress_cb.set_description(f"{row['distance_km']:.2f} km - {row['name']}")
        else:
            print(f"{row['distance_km']:.2f} km - {row['name']}")

        file_name = row["file_name"]

        if not file_name.endswith(".fit.gz") and not file_name.endswith(".fit"):
            print(
                f"⏩Skipping {file_name} ({row['name']}) because it is not a FIT file"
            )
            continue

        run = RunningActivities.from_computed_file(
            file_name, segment_width, progress_cb=progress_cb
        )
        run._data["name"] = row["name"]
        running_activities.append(run)

    return running_activities


def create_dataset(
    activities=None,
    min_distance=30,
    max_distance=100,
    segment_width=500,
    progress_cb: ProgressCallback | None = None,
) -> pd.DataFrame:
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
    """Split the dataset into a training and a test set and keeping activity_id together."""
    activity_ids = data["activity_id"].unique()
    train_ids, test_ids = train_test_split(
        activity_ids, test_size=test_size, random_state=random_state
    )
    train_data = data.loc[data["activity_id"].isin(train_ids)]
    test_data = data.loc[data["activity_id"].isin(test_ids)]
    return train_data, test_data


def _window_sum(s: pd.Series, k: int, direction: str = "front") -> pd.Series:
    """Somme sur les k segments précédents ('back') ou suivants ('front'),
    segment courant exclu. Aux bords, on somme ce qui est disponible."""
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


def _cumsum(s: pd.Series, direction: str = "forward") -> pd.Series:
    """Cumulative sum of a series, either forward or backward."""
    if direction == "forward":
        return s.cumsum()
    return s.transform(lambda values: values.iloc[::-1].cumsum().iloc[::-1])


def build_features(
    dataset: pd.DataFrame, windows_km: tuple = WINDOWS_KM
) -> pd.DataFrame:
    """Build features for the dataset."""
    x = dataset.copy()
    x_grp = x.groupby("activity_id")

    # => difference in altitude between two consecutive points
    x["altitude_drop"] = x_grp["altitude"].diff().fillna(0)
    x["altitude_drop_pos"] = x["altitude_drop"].clip(lower=0)
    x["altitude_drop_neg"] = x["altitude_drop"].clip(upper=0)
    # cumulative of each variable for "done" and "remaining" features
    x_grp = x.groupby("activity_id")
    col_features = ("altitude_drop_pos", "altitude_drop_neg", "distance")
    for feature in col_features:
        x[f"{feature}_done"] = _cumsum(x_grp[feature], direction="forward")
        x[f"{feature}_remaining"] = _cumsum(x_grp[feature], direction="backward")
        x = x.merge(
            x.groupby("activity_id")[feature]
            .sum()
            .reset_index()
            .rename(columns={feature: f"{feature}_total"})
        )

    # => ratio of each variable for "done" and "remaining" features
    # in order to remove the effect of the total value of each variable for each activity
    for feature in col_features:
        for suffix in ("done", "remaining"):
            col = f"{feature}_{suffix}"
            x[f"{col}_ratio"] = x[col] / x[f"{feature}_total"]

    # => sum of each variable for the previous and next k segments
    x_grp = x.groupby("activity_id")
    segment_width = dataset["segment"].diff().max()
    for feature in col_features:
        for k in [int(w * 1000 / segment_width) for w in WINDOWS_KM]:
            x[f"{feature}_back_{k}"] = _window_sum(x_grp[feature], k, direction="back")
            x[f"{feature}_front_{k}"] = _window_sum(
                x_grp[feature], k, direction="front"
            )

    # => Minetti energy cost for running and walking
    x["slope"] = x["altitude_drop"] / x["distance"]
    x["walking_cost"] = cost_of_walking(x["slope"]) / cost_of_walking(0)
    x["running_cost"] = cost_of_running(x["slope"]) / cost_of_running(0)

    # TODO : ravito

    return x
