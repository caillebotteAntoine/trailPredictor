"""Tests for :class:`TrailPredictor.modeling.config.ModelConfig`."""

import pytest
from pydantic import ValidationError

from TrailPredictor.modeling.config import KEY_LENGTH, ModelConfig


def test_defaults() -> None:
    config = ModelConfig()

    assert (config.seed, config.n_splits) == (42, 5)


def test_is_frozen() -> None:
    config = ModelConfig()

    with pytest.raises(ValidationError, match="frozen"):
        config.n_splits = 3


def test_unknown_field_raises() -> None:
    with pytest.raises(ValidationError, match="n_split"):
        ModelConfig(n_split=3)


@pytest.mark.parametrize("n_splits", [0, 1])
def test_too_few_splits_raises(n_splits: int) -> None:
    with pytest.raises(ValidationError, match="n_splits"):
        ModelConfig(n_splits=n_splits)


def test_wrong_type_raises() -> None:
    with pytest.raises(ValidationError, match="seed"):
        ModelConfig(seed="not a seed")


def test_key_is_stable_and_short() -> None:
    key = ModelConfig().key

    assert key == ModelConfig(seed=42, n_splits=5).key
    assert len(key) == KEY_LENGTH


def test_key_depends_on_every_field() -> None:
    keys = {
        ModelConfig().key,
        ModelConfig(seed=0).key,
        ModelConfig(n_splits=3).key,
    }

    assert len(keys) == 3
