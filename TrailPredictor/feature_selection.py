"""Ranking, filtering and selection of the dataset features.

UI-independent helpers behind the feature explorer: the correlation of each
feature with the target, the filters, the selection presets and the merge of
the ticked boxes into the selection.
"""

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum

import numpy as np
import pandas as pd

from TrailPredictor.dataset import Dataset

# The Spearman correlation is flipped for pace-like targets, so that it is
# always expressed with respect to the speed (+: the feature goes with a
# faster running).
PACE_SIGN = -1.0
SPEED_SIGN = 1.0
PACE_SUFFIX = "pace"

CORRELATION_COLUMNS = ["name", "spearman", "missing_pct", "group"]
PERCENT = 100.0
MISSING_PCT_DECIMALS = 2
_TARGET_COLUMN = "__target__"


class FeatureGroup(StrEnum):
    """Family of a feature, deduced from its name."""

    PAST = "past"
    UPCOMING = "upcoming"
    CUMULATIVE = "cumulative"
    RATIO = "ratio"
    OTHER = "other"


class Preset(StrEnum):
    """Predefined selections picked among the filtered features."""

    TOP_5 = "top_5"
    TOP_10 = "top_10"
    TOP_20 = "top_20"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ALL = "all"


def feature_group(name: str) -> FeatureGroup:
    """Return the family of a feature from its name.

    Parameters
    ----------
    name : str
        Feature name, e.g. ``"altitude_drop_back_1km"``.

    Returns
    -------
    FeatureGroup
        ``PAST`` / ``UPCOMING`` for the back / front windows, ``CUMULATIVE``
        for the ``_done``, ``_remaining`` and ``_total`` sums, ``RATIO`` for
        the ``_ratio`` features and ``OTHER`` otherwise.

    Examples
    --------
    >>> feature_group("altitude_drop_front_2km")
    <FeatureGroup.UPCOMING: 'upcoming'>
    """
    if "_back_" in name:
        return FeatureGroup.PAST
    if "_front_" in name:
        return FeatureGroup.UPCOMING
    if name.endswith(("_done", "_remaining", "_total")):
        return FeatureGroup.CUMULATIVE
    if name.endswith("_ratio"):
        return FeatureGroup.RATIO
    return FeatureGroup.OTHER


def correlation_table(dataset: Dataset) -> pd.DataFrame:
    """Compute the Spearman correlation of each numeric feature with the target.

    The sign is flipped for pace-like targets, so a positive correlation always
    means a faster running.

    Parameters
    ----------
    dataset : Dataset
        Dataset whose target (``dataset.target_name``) the features are
        correlated with. It is not modified.

    Returns
    -------
    pandas.DataFrame
        Columns ``name``, ``spearman``, ``missing_pct`` (percentage of missing
        values) and ``group`` (`FeatureGroup` value), one row per numeric
        feature, sorted by decreasing absolute correlation (NaN last).
    """
    data = dataset.data
    names = [
        name for name in dataset.features if pd.api.types.is_numeric_dtype(data[name])
    ]
    target = (
        pd.to_numeric(dataset.target, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    frame = (
        data[names]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .assign(**{_TARGET_COLUMN: target})
    )
    sign = PACE_SIGN if dataset.target_name.endswith(PACE_SUFFIX) else SPEED_SIGN
    rho = frame.corr(method="spearman")[_TARGET_COLUMN].drop(_TARGET_COLUMN) * sign

    table = pd.DataFrame(
        {
            "name": rho.index,
            "spearman": rho.to_numpy(),
            "missing_pct": (data[names].isna().mean() * PERCENT)
            .round(MISSING_PCT_DECIMALS)
            .to_numpy(),
            "group": [feature_group(name).value for name in rho.index],
        },
        columns=CORRELATION_COLUMNS,
    )
    order = table["spearman"].abs().sort_values(ascending=False, na_position="last")
    return table.loc[order.index].reset_index(drop=True)


def apply_filters(
    correlations: pd.DataFrame,
    search: str = "",
    groups: Sequence[str] = (),
    min_abs: float = 0.0,
    max_missing: float = PERCENT,
) -> pd.DataFrame:
    """Keep the features that pass every filter, in the same order.

    Parameters
    ----------
    correlations : pandas.DataFrame
        Table returned by `correlation_table`.
    search : str, optional
        Case-insensitive substring of the feature name. Default is ``""``
        (no filter).
    groups : sequence of str, optional
        Allowed `FeatureGroup` values; all groups when empty (default).
    min_abs : float, optional
        Minimum absolute Spearman correlation; no filter when 0 (default),
        so the features with a NaN correlation are kept.
    max_missing : float, optional
        Maximum percentage of missing values. Default is 100.

    Returns
    -------
    pandas.DataFrame
        Filtered table with a fresh index.
    """
    mask = correlations["name"].str.contains(search, case=False, na=False, regex=False)
    if groups:
        mask &= correlations["group"].isin(groups)
    mask &= correlations["missing_pct"] <= max_missing
    if min_abs > 0:
        mask &= correlations["spearman"].abs() >= min_abs
    return correlations.loc[mask].reset_index(drop=True)


def _ranked(pool: pd.DataFrame) -> pd.DataFrame:
    return pool.dropna(subset=["spearman"])


_PRESETS: Mapping[Preset, Callable[[pd.DataFrame], pd.Series]] = {
    Preset.TOP_5: lambda pool: _ranked(pool).head(5)["name"],
    Preset.TOP_10: lambda pool: _ranked(pool).head(10)["name"],
    Preset.TOP_20: lambda pool: _ranked(pool).head(20)["name"],
    Preset.POSITIVE: lambda pool: _ranked(pool).loc[lambda d: d["spearman"] > 0, "name"],
    Preset.NEGATIVE: lambda pool: _ranked(pool).loc[lambda d: d["spearman"] < 0, "name"],
    Preset.ALL: lambda pool: pool["name"],
}


def apply_preset(preset: Preset, pool: pd.DataFrame) -> list[str]:
    """Return the features of a preset, picked among the filtered features.

    Parameters
    ----------
    preset : Preset
        Predefined selection.
    pool : pandas.DataFrame
        Filtered table returned by `apply_filters`, sorted by decreasing
        absolute correlation.

    Returns
    -------
    list of str
        Names of the features of the preset, in the order of `pool`. The
        ranked presets ignore the features with a NaN correlation.
    """
    return _PRESETS[preset](pool).tolist()


def merge_edits(current: Sequence[str], flags: Mapping[str, bool]) -> list[str]:
    """Apply the ticked boxes of the table to the selection.

    Parameters
    ----------
    current : sequence of str
        Currently selected features, in selection order.
    flags : Mapping[str, bool]
        Tick state of each feature shown in the table.

    Returns
    -------
    list of str
        Updated selection. Features missing from `flags` (hidden by a
        filter) are unchanged, the order is kept and new features are
        appended.
    """
    updated = [name for name in current if flags.get(name, True)]
    updated += [name for name, on in flags.items() if on and name not in updated]
    return updated
