import numpy as np
import pytest
from pytest import approx

from desdeo.solver.ASF import SimpleASF
from desdeo.solver.Solver import (
    ASFSolver,
    IdealAndNadirPointSolver,
    WeightingMethodSolver,
)


@pytest.fixture
def WeightedCylinderSolver(CylinderProblem):
    return WeightingMethodSolver(CylinderProblem)


@pytest.fixture
def SimpleASFCylinderSolver(CylinderProblem):
    return ASFSolver(CylinderProblem)


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
    expected = [1982.2033717615695, 503.733320193871, 7456.610960526498]

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
            [1982.2033717615695, 503.733320193871, 7456.610960526498],
        )
    )

    assert np.all(np.isclose(res, expected))


def test_weighting_evaluator_single_decision_vector(
    WeightedCylinderSolver, cylinder_good_decision_vectors
):
    weights = np.ones(3)
    WeightedCylinderSolver.weights = weights
    res = WeightedCylinderSolver._evaluator(cylinder_good_decision_vectors[0])

    assert res == approx(1982.2033717615695)


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
    assert np.all(
        np.isclose(variables, expected_variables, rtol=0.0, atol=1.0e-1)
    )

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


def test_asf_evaluator(
    SimpleASFCylinderSolver,
    cylinder_good_decision_vectors,
    cylinder_bad_decision_vectors,
):
    solver = SimpleASFCylinderSolver
    weights = np.array([1.0, 1.0, 1.0])
    solver.asf = SimpleASF(weights)
    solver.reference_point = np.array([500, 500, 500])

    res_good = solver._evaluator(cylinder_good_decision_vectors)

    res_bad = solver._evaluator(cylinder_bad_decision_vectors)

    assert np.all(res_good != np.inf)
    assert np.all(res_bad == np.inf)


def test_asf_evaluator_zero_weights(
    SimpleASFCylinderSolver,
    cylinder_good_decision_vectors,
    cylinder_bad_decision_vectors,
):
    solver = SimpleASFCylinderSolver
    weights = np.zeros(3)
    solver.asf = SimpleASF(weights)
    solver.reference_point = np.array([500, 500, 500])

    res_good = solver._evaluator(cylinder_good_decision_vectors)
    res_bad = solver._evaluator(cylinder_bad_decision_vectors)

    assert np.all(res_good == approx(0))
    assert np.all(res_bad == np.inf)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_asf_solve_ones_weights(SimpleASFCylinderSolver):
    weights = np.ones(3)
    solver = SimpleASFCylinderSolver
    solver.asf = SimpleASF(weights)
    reference_point = np.array([500, 500, 500])

    (variables, (objectives, constraints)) = solver.solve(reference_point)

    expected_variables = np.array([5.0, 10.0])

    assert np.all(
        np.isclose(variables, expected_variables, rtol=0.0, atol=1.0e-1)
    )
    assert np.all(np.greater_equal(constraints, 0))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_asf_solve_reference_height(SimpleASFCylinderSolver):
    """Ignore all other objectives but the hieght difference. Results should
    therefore be the optimal height according to the 3rd objective which is
    15, regardless of the reference value for the objective.

    """
    weights = np.array([1, 1, 1])
    solver = SimpleASFCylinderSolver
    solver.asf = SimpleASF(weights)
    reference_point = np.array([np.nan, np.nan, 6])

    (variables, (objectives, constraints)) = solver.solve(reference_point)

    assert objectives[0][2] == approx(0, abs=1e-3)
    # Solution should always be feasible
    assert np.all(np.greater_equal(constraints, 0))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_asf_solve_reference_extreme_area_and_volume(SimpleASFCylinderSolver):
    """Test a reference point residing very close to the pareto front in the
    2nd objective space. The 1st objective is set to uniquely to specify the
    variables to be r=12.5 and h=25. The third objective is ignored.

    """
    weights = np.array([1, 1, 1])
    solver = SimpleASFCylinderSolver
    solver.asf = SimpleASF(weights)
    reference_point = np.array([12271.8, -2945.25, np.nan])

    (variables, (objectives, constraints)) = solver.solve(reference_point)

    assert objectives[0][0] == approx(reference_point[0], abs=5e-2)
    assert objectives[0][1] == approx(reference_point[1], abs=5e-2)
    assert variables[0] == approx(12.5, abs=1e-3)
    assert variables[1] == approx(25.0, abs=1e-3)
    assert np.all(np.greater_equal(constraints, 0))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_asf_solve_reference_pareto(SimpleASFCylinderSolver):
    """Test a reference point residing very close to the pareto front.
    The resulting solution should be close to this point.

    """
    weights = np.array([1, 1, 1])
    solver = SimpleASFCylinderSolver
    solver.asf = SimpleASF(weights)
    reference_point = np.array([785.398, -471.239, 5.0])

    (variables, (objectives, constraints)) = solver.solve(reference_point)

    assert objectives[0][0] == approx(reference_point[0], abs=5e-2)
    assert objectives[0][1] == approx(reference_point[1], abs=5e-2)
    assert variables[0] == approx(5.0, abs=1e-3)
    assert variables[1] == approx(10.0, abs=1e-3)
    assert np.all(np.greater_equal(constraints, 0))


def test_ideal_and_nadir_point_evaluator(CylinderProblem):
    solver = IdealAndNadirPointSolver(CylinderProblem)
    decision_vector = np.array([7, 15])

    res_0 = solver._evaluator(decision_vector, 0)
    res_1 = solver._evaluator(decision_vector, 1)
    res_2 = solver._evaluator(decision_vector, 2)

    assert res_0 == approx(2309.070600388498)
    assert res_1 == approx(-967.6105373056563)
    assert res_2 == approx(0.0)

    bad_decision_vector = np.array([10, 5])
    res_bads = np.array(
        [solver._evaluator(bad_decision_vector, ind) for ind in range(3)]
    )

    assert np.all(res_bads == np.inf)


@pytest.mark.snipe
def test_ideal_and_nadir_point_solver(CylinderProblem):
    solver = IdealAndNadirPointSolver(CylinderProblem)
    res, nadir = solver.solve()
    print(nadir)
    expected = np.array([785.476703213788, -2945.321652556771, 0.0])

    assert np.all(np.isclose(res, expected, rtol=0, atol=0.5))
