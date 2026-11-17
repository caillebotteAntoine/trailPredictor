"""Tests for :class:`TrailPredictor.plot.dataset.PlotDataset`."""

import warnings

import pandas as pd
import plotly.graph_objects as go
import pytest

from TrailPredictor.dataset import Dataset

PREDICTION_FACTOR = 1.1  # predictions 10 % above the target


@pytest.fixture
def predicted_dataset(dataset: pd.DataFrame) -> Dataset:
    ds = Dataset(dataset)
    ds.prediction = ds.target * PREDICTION_FACTOR
    return ds


def _unity_line(fig: go.Figure) -> go.Scatter:
    (line,) = [trace for trace in fig.data if trace.name == "Unity line"]
    return line


def test_vs_prediction_emits_no_deprecation_warning(
    predicted_dataset: Dataset,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predicted_dataset.plot.vs_prediction()

    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_vs_prediction_unity_line_spans_predictions(
    predicted_dataset: Dataset,
) -> None:
    line = _unity_line(predicted_dataset.plot.vs_prediction())

    prediction = predicted_dataset.prediction
    expected = [prediction.min(), prediction.max()]
    assert isinstance(line, go.Scatter)
    assert line.mode == "lines"
    assert list(line.x) == pytest.approx(expected)
    assert list(line.y) == pytest.approx(expected)


def test_vs_prediction_without_prediction_raises(dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Prediction column"):
        Dataset(dataset).plot.vs_prediction()
