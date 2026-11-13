"""Tests for :mod:`TrailPredictor.minetti`."""

import numpy as np
import pytest

from TrailPredictor.minetti import cost_of_running, cost_of_walking


@pytest.mark.parametrize(
    ("cost", "base", "coefficients"),
    [
        (cost_of_running, 3.6, (19.5, 46.3, -43.3, -30.4, 155.4)),
        (cost_of_walking, 2.5, (19.6, 51.9, -76.8, -58.7, 280.5)),
    ],
)
def test_cost_matches_polynomial(cost, base, coefficients):
    grade = 0.2
    expected = base + sum(
        coefficient * grade**power for power, coefficient in enumerate(coefficients, 1)
    )

    assert cost(grade) == pytest.approx(expected)


@pytest.mark.parametrize("cost", [cost_of_running, cost_of_walking])
def test_cost_clips_to_valid_grade_range(cost):
    assert cost(-1) == pytest.approx(cost(-0.45))
    assert cost(1) == pytest.approx(cost(0.45))


def test_cost_supports_array_like_input():
    grades = [-0.5, 0, 0.5]

    result = cost_of_running(grades)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(
        result,
        [cost_of_running(-0.45), cost_of_running(0), cost_of_running(0.45)],
    )
