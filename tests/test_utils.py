"""Tests for :mod:`TrailPredictor.utils`."""

from pathlib import Path

import pytest

from TrailPredictor import loading, utils
from TrailPredictor.utils import DATA_DIR_ENV, SOURCE_ROOT, resolve_data_dir

EXECUTABLE = Path("C:/Apps/TrailPredictor/TrailPredictor.exe")


def test_sources_use_the_repository_data() -> None:
    assert resolve_data_dir({}, frozen=False) == SOURCE_ROOT / "data"


def test_executable_uses_the_data_next_to_it() -> None:
    data_dir = resolve_data_dir({}, frozen=True, executable=EXECUTABLE)

    assert data_dir == Path("C:/Apps/TrailPredictor/data")


@pytest.mark.parametrize("frozen", [False, True])
def test_environment_variable_takes_precedence(tmp_path: Path, frozen: bool) -> None:
    data_dir = resolve_data_dir(
        {DATA_DIR_ENV: str(tmp_path)}, frozen=frozen, executable=EXECUTABLE
    )

    assert data_dir == tmp_path


def test_environment_variable_expands_the_home_directory() -> None:
    data_dir = resolve_data_dir({DATA_DIR_ENV: "~/strava"}, frozen=False)

    assert data_dir == Path.home() / "strava"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_environment_variable_is_ignored(value: str) -> None:
    assert resolve_data_dir({DATA_DIR_ENV: value}, frozen=False) == SOURCE_ROOT / "data"


def test_derived_directories_and_activity_summary() -> None:
    assert utils.COMPUTED_DIR == utils.DATA_DIR / "computed"
    assert utils.MODELS_DIR == utils.COMPUTED_DIR / "models"
    assert loading.ACTIVITIES_FILE == utils.DATA_DIR / "activities.csv"
