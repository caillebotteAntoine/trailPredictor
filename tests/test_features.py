"""Tests for :func:`TrailPredictor.features.export_activities`."""

import contextlib
import io
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from TrailPredictor import features
from TrailPredictor.activities import RunningActivities


@pytest.fixture
def summary() -> pd.DataFrame:
    """Two runs within 30-100 km (one without FIT file) and a ride."""
    return pd.DataFrame(
        {
            "type": ["run", "run", "ride"],
            "distance_km": [42.0, 50.0, 60.0],
            "name": ["🏃 Marathon 🌞", "⏩ Manual entry", "Ride"],
            "file_name": [
                "activities/1.fit.gz",
                "activities/2.gpx",
                "activities/3.fit",
            ],
        }
    )


@pytest.fixture
def cp1252_stdout() -> io.TextIOWrapper:
    """Return a cp1252 stream, as the stdout of a redirected Windows console.

    Tests redirect stdout to it in their body: pytest resets ``sys.stdout``
    between the setup and the call of a test.
    """
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


@pytest.fixture
def loaded_files(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the files loaded by ``from_computed_file`` instead of reading them."""
    files: list[str] = []

    def fake_load(
        _cls: type, file_path: str, *_args: object, **_kwargs: object
    ) -> SimpleNamespace:
        files.append(file_path)
        return SimpleNamespace(name=None)  # the loader only sets the name

    monkeypatch.setattr(RunningActivities, "from_computed_file", classmethod(fake_load))
    return files


def test_export_activities_skips_non_fit_files(
    summary: pd.DataFrame, loaded_files: list[str]
) -> None:
    runs = features.export_activities(summary)

    assert loaded_files == ["activities/1.fit.gz"]
    assert [run.name for run in runs] == ["🏃 Marathon 🌞"]


def test_export_activities_logs_instead_of_printing(
    summary: pd.DataFrame,
    loaded_files: list[str],
    cp1252_stdout: io.TextIOWrapper,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger=features.__name__),
        contextlib.redirect_stdout(cp1252_stdout),
    ):
        features.export_activities(summary)

    cp1252_stdout.flush()
    assert not cp1252_stdout.buffer.getvalue()  # type: ignore[attr-defined]
    assert "🏃 Marathon 🌞" in caplog.text
    assert "activities/2.gpx" in caplog.text
    assert len(loaded_files) == 1
