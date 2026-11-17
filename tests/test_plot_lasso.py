"""Tests for :class:`TrailPredictor.plot.PlotLasso`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.lasso import LassoConfig, LassoPath, ModelingLasso
from TrailPredictor.plot import PlotLasso
from TrailPredictor.plot._common import CATEGORICAL_COLORS

N_FEATURES = 12  # more than the palette: highlighted features change dash
N_ALPHAS = 4
N_FOLDS = 2


@pytest.fixture
def model(lasso_data: pd.DataFrame, tmp_path: Path) -> ModelingLasso:
    config = LassoConfig(n_splits=2, inner_n_splits=N_FOLDS, n_alphas=N_ALPHAS)
    return ModelingLasso(Dataset(lasso_data), config, models_dir=tmp_path)


@pytest.fixture
def toy_path() -> LassoPath:
    """Feature ``f{k}`` has coefficient ``k + 1`` at the two smallest alphas."""
    alphas = np.logspace(0, -3, N_ALPHAS)
    weights = np.arange(1, N_FEATURES + 1, dtype=np.float64)
    coefs = np.outer([0.0, 0.0, 1.0, 1.0], weights)
    # the 3 smallest features are inactive at the second alpha
    coefs[1, 3:] = weights[3:]
    return LassoPath(
        coef_path=pd.DataFrame(
            coefs, index=alphas, columns=[f"f{k}" for k in range(N_FEATURES)]
        ),
        mse_path=np.array([[5.0, 5.2], [2.0, 2.2], [1.0, 3.0], [1.9, 2.1]]),
        alpha=float(alphas[2]),
    )


@pytest.fixture
def plot(model: ModelingLasso, toy_path: LassoPath) -> PlotLasso:
    plot = PlotLasso(model)
    plot.__dict__["path"] = toy_path  # skip the LassoCV fit: set the cached path
    return plot


def _traces(fig, name: str) -> list:
    return [trace for trace in fig.data if trace.name == name]


def test_plot_shares_model_settings(model: ModelingLasso) -> None:
    plot = model.plot

    assert isinstance(plot, PlotLasso)
    assert plot.config is model.config
    assert plot.dataset is model.dataset
    assert plot.result_path == model.result_path


def test_path_is_fitted_once(
    model: ModelingLasso, monkeypatch: pytest.MonkeyPatch
) -> None:
    plot = PlotLasso(model)
    calls: list[LassoPath] = []

    def spy_fit() -> LassoPath:
        calls.append(ModelingLasso.fit_regularization_path(plot))
        return calls[-1]

    monkeypatch.setattr(plot, "fit_regularization_path", spy_fit)

    first = plot.path
    second = plot.path

    assert second is first
    assert len(calls) == 1
    assert first.coef_path.shape[0] == N_ALPHAS


def test_highlights_active_features_at_alpha_1se(
    plot: PlotLasso, toy_path: LassoPath
) -> None:
    fig = plot.regularization_path()

    # threshold 1.0 + 2.0 / sqrt(2) > 2.1: the 1-SE rule picks the 2nd alpha
    assert toy_path.alpha_1se == pytest.approx(toy_path.alphas[1])
    highlighted = [f"f{k}" for k in range(N_FEATURES - 1, 2, -1)]
    assert [trace.name for trace in fig.data if trace.name in highlighted] == (
        highlighted
    )
    others = _traces(fig, "Autres variables")
    assert len(others) == 3
    assert [trace.showlegend for trace in others] == [True, False, False]


def test_highlight_colors_and_dashes(plot: PlotLasso) -> None:
    fig = plot.regularization_path(n_highlight=N_FEATURES)

    first, wrapped = _traces(fig, f"f{N_FEATURES - 1}")[0], _traces(fig, "f1")[0]
    assert (first.line.color, first.line.dash) == (CATEGORICAL_COLORS[0], "solid")
    # f1 has rank 10 = len(CATEGORICAL_COLORS): first colour again, next dash
    assert (wrapped.line.color, wrapped.line.dash) == (CATEGORICAL_COLORS[0], "dash")


@pytest.mark.parametrize("n_highlight", [0, 2])
def test_n_highlight(plot: PlotLasso, n_highlight: int) -> None:
    fig = plot.regularization_path(n_highlight=n_highlight)

    assert len(_traces(fig, "Autres variables")) == N_FEATURES - n_highlight


def test_negative_n_highlight_raises(plot: PlotLasso) -> None:
    with pytest.raises(ValueError, match="n_highlight"):
        plot.regularization_path(n_highlight=-1)


def test_cv_error_and_alpha_markers(plot: PlotLasso, toy_path: LassoPath) -> None:
    fig = plot.regularization_path()

    mean_trace = _traces(fig, "Erreur CV moyenne")[0]
    np.testing.assert_allclose(mean_trace.y, toy_path.mse_mean)
    np.testing.assert_allclose(mean_trace.x, toy_path.alphas)
    selected = _traces(fig, f"α retenu (CV) = {toy_path.alpha:.3g}")
    one_se = _traces(fig, f"α 1 erreur-type = {toy_path.alpha_1se:.3g}")
    # one vertical line per row, a single legend entry
    assert [trace.yaxis for trace in selected] == ["y", "y2"]
    assert [trace.showlegend for trace in one_se] == [True, False]
    np.testing.assert_allclose(selected[0].x, [toy_path.alpha] * 2)
    np.testing.assert_allclose(one_se[1].x, [toy_path.alpha_1se] * 2)


def test_log_alpha_axis(plot: PlotLasso) -> None:
    fig = plot.regularization_path()

    assert fig.layout.xaxis.type == "log"
    assert fig.layout.xaxis2.type == "log"
