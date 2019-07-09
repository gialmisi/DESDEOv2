import numpy as np
import pytest
from pytest import approx

from desdeo.solver.Solver import WeightingMethodSolver


@pytest.fixture
def WeightedCylinderSolver(CylinderProblem):
    return WeightingMethodSolver(CylinderProblem)


@pytest.fixture
def cylinder_good_decision_vectors():
    """Inside bounds and do not break any constraints."""
    return np.array([[7.5, 18], [5.1, 13.4], [11.5, 24.12]])


@pytest.fixture
def cylinder_bad_decision_vectors():
    """All constraints broken"""
    return np.array([[18, 7.5], [13.4, 5.1], [24.12, 11.5]])


def test_weighting_evaluator_zero_weights(
    WeightedCylinderSolver, cylinder_good_decision_vectors
):
    weights_zeros = np.zeros(3)
    WeightedCylinderSolver.weights = weights_zeros
    sums = WeightedCylinderSolver._evaluator(cylinder_good_decision_vectors)
    assert np.all(sums == approx(0.0))


def test_weighting_evaluator_ones_weights(
    WeightedCylinderSolver, cylinder_good_decision_vectors
):
    weights_ones = np.ones(3)
    WeightedCylinderSolver.weights = weights_ones
    res = WeightedCylinderSolver._evaluator(cylinder_good_decision_vectors)
    expected = [2560.4333882308138, 796.446204086523, 8644.68090103197]

    assert np.all(np.isclose(res, expected))


def test_weighting_evaluator_even_weights(
    WeightedCylinderSolver, cylinder_good_decision_vectors
):
    weights_even = np.array([0.33, 0.33, 0.33])
    WeightedCylinderSolver.weights = weights_even
    res = WeightedCylinderSolver._evaluator(cylinder_good_decision_vectors)

    expected = list(
        map(
            lambda x: x * 0.33,
            [2560.4333882308138, 796.446204086523, 8644.68090103197],
        )
    )

    assert np.all(np.isclose(res, expected))


def test_weighting_evaluator_single_decision_vector(
    WeightedCylinderSolver, cylinder_good_decision_vectors
):
    weights = np.ones(3)
    WeightedCylinderSolver.weights = weights
    res = WeightedCylinderSolver._evaluator(cylinder_good_decision_vectors[0])

    assert res == approx(2560.4333882308138)


def test_weighting_evaluator_broken_constraints(
    WeightedCylinderSolver, cylinder_bad_decision_vectors
):
    weights = np.ones(3)
    WeightedCylinderSolver.weights = weights
    res = WeightedCylinderSolver._evaluator(cylinder_bad_decision_vectors)
    expected = [np.inf, np.inf, np.inf]

    assert np.all(np.isclose(res, expected))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_weighting_solve_ones_weights(WeightedCylinderSolver):
    # Suppress RuntimeWarnings, most commonly produced by infinities
    weights = np.ones(3)
    solver = WeightedCylinderSolver
    (variables, (objectives, constraints)) = solver.solve(weights)

    expected_variables = np.array([5.0, 10.0])
    # Note the quire high absolute tolerance and no relative tolerance.
    # The results vary quire a bit, so a high tolerance was chosen.
    assert np.all(np.isclose(variables,
                             expected_variables,
                             rtol=0.0,
                             atol=1.e-1))

    assert np.all(np.greater_equal(constraints, 0.0))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_weighting_solve_single_weight(WeightedCylinderSolver):
    # Suppress RuntimeWarnings, most commonly produced by infinities
    # Prefer the third objective
    weights = np.array([0.0, 0.0, 1.0])
    solver = WeightedCylinderSolver
    (variables, (objectives, constraints)) = solver.solve(weights)

    # Height should be 15, radius can be whatever since the 3rd objective does
    # not care for it
    assert variables[1] == approx(15.0)

    # Constraints should still hold!
    assert np.all(np.greater_equal(constraints, 0.0))
