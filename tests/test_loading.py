"""Tests for :mod:`TrailPredictor.loading`."""

import pandas as pd
import pytest

from TrailPredictor.loading import (
    FRENCH_MONTHS,
    UnknownActivityError,
    activity_file_name,
    parse_strava_date,
    running_activities,
)


@pytest.fixture
def summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [11, 12, 13, 13],
            "type": ["run", "ride", "run", "run"],
            "file_name": [
                "activities/11.fit.gz",
                "activities/12.fit",
                "a.fit",
                "b.fit",
            ],
        },
        index=[5, 6, 7, 8],
    )


# --- running_activities


def test_running_activities_keeps_runs_and_index(summary: pd.DataFrame) -> None:
    runs = running_activities(summary)

    assert runs["id"].tolist() == [11, 13, 13]
    assert runs.index.tolist() == [5, 7, 8]


def test_running_activities_returns_a_copy(summary: pd.DataFrame) -> None:
    before = summary.copy()

    runs = running_activities(summary)
    runs.loc[:, "file_name"] = "changed"

    pd.testing.assert_frame_equal(summary, before)


def test_running_activities_empty() -> None:
    empty = pd.DataFrame({"type": pd.Series([], dtype="str")})

    assert running_activities(empty).empty


# --- parse_strava_date


@pytest.mark.parametrize(
    ("french", "month"),
    [(french, month) for month, french in enumerate(FRENCH_MONTHS, start=1)],
)
def test_parse_strava_date_every_month(french: str, month: int) -> None:
    parsed = parse_strava_date(f"2 {french} 2026, 07:30:00")

    assert parsed == pd.Timestamp(2026, month, 2, 7, 30)


@pytest.mark.parametrize("value", ["not a date", None, float("nan"), ""])
def test_parse_strava_date_invalid_gives_nat(value: object) -> None:
    assert pd.isna(parse_strava_date(value))


# --- activity_file_name


def test_activity_file_name(summary: pd.DataFrame) -> None:
    assert activity_file_name(summary, 11) == "activities/11.fit.gz"


def test_activity_file_name_duplicate_id_takes_first(summary: pd.DataFrame) -> None:
    assert activity_file_name(summary, 13) == "a.fit"


def test_activity_file_name_unknown_id_raises(summary: pd.DataFrame) -> None:
    with pytest.raises(UnknownActivityError, match="99"):
        activity_file_name(summary, 99)
