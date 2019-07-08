import numpy as np
import pytest
from pytest import approx

from desdeo.solver.Solver import WeightingMethodSolver


@pytest.fixture
def WeightedCylinderProblem(CylinderProblem):
    return WeightingMethodSolver(CylinderProblem)


@pytest.fixture
def cylinder_good_decision_vectors():
    """Inside bounds and do not break any constraints."""
    return np.array([[7.5, 18], [5.1, 13.4], [11.5, 24.12]])


@pytest.fixture
def cylinder_bad_decision_vectors():
    """All constraints broken"""
    return np.array([[18, 7.5], [13.4, 5.1], [24.12, 11.5]])


def test_evaluator_zero_weights(
    WeightedCylinderProblem, cylinder_good_decision_vectors
):
    weights_zeros = np.zeros(3)
    WeightedCylinderProblem.weights = weights_zeros
    sums = WeightedCylinderProblem._evaluator(cylinder_good_decision_vectors)
    assert np.all(sums == approx(0.0))


def test_evaluator_ones_weights(
    WeightedCylinderProblem, cylinder_good_decision_vectors
):
    weights_ones = np.ones(3)
    WeightedCylinderProblem.weights = weights_ones
    res = WeightedCylinderProblem._evaluator(cylinder_good_decision_vectors)
    expected = [2560.4333882308138, 796.446204086523, 8644.68090103197]

    assert np.all(np.isclose(res, expected))


def test_evaluator_even_weights(
    WeightedCylinderProblem, cylinder_good_decision_vectors
):
    weights_even = np.array([0.33, 0.33, 0.33])
    WeightedCylinderProblem.weights = weights_even
    res = WeightedCylinderProblem._evaluator(cylinder_good_decision_vectors)

    expected = list(
        map(
            lambda x: x * 0.33,
            [2560.4333882308138, 796.446204086523, 8644.68090103197],
        )
    )

    assert np.all(np.isclose(res, expected))


def test_evaluator_single_decision_vector(
    WeightedCylinderProblem, cylinder_good_decision_vectors
):
    weights = np.ones(3)
    WeightedCylinderProblem.weights = weights
    res = WeightedCylinderProblem._evaluator(cylinder_good_decision_vectors[0])

    assert res == approx(2560.4333882308138)


def test_evaluator_broken_constraints(
    WeightedCylinderProblem, cylinder_bad_decision_vectors
):
    weights = np.ones(3)
    WeightedCylinderProblem.weights = weights
    res = WeightedCylinderProblem._evaluator(cylinder_bad_decision_vectors)
    expected = [np.inf, np.inf, np.inf]

    assert np.all(np.isclose(res, expected))
