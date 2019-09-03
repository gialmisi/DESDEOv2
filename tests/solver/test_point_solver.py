import numpy as np
import pytest
from pytest import approx

from desdeov2.solver.NumericalMethods import ScipyDE
from desdeov2.solver.PointSolver import IdealAndNadirPointSolver


@pytest.fixture
def Scipyde_method():
    method = ScipyDE(
        {"tol": 0.000001, "popsize": 10, "maxiter": 50000, "polish": True}
    )
    return method


def test_ideal_and_nadir_point_evaluator(CylinderProblem, Scipyde_method):
    solver = IdealAndNadirPointSolver(CylinderProblem, Scipyde_method)
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


def test_ideal_and_nadir_point_solver(CylinderProblem, Scipyde_method):
    solver = IdealAndNadirPointSolver(CylinderProblem, Scipyde_method)
    ideal, nadir = solver.solve()
    expected_ideal = np.array([785.476703213788, -2945.321652556771, 0.0])

    assert np.all(np.isclose(ideal, expected_ideal, rtol=0, atol=0.5))
    assert np.all(np.greater(nadir, ideal))


def test_ideal_and_nadir_point_solver_river(
    RiverPollutionProblem, Scipyde_method
):
    solver = IdealAndNadirPointSolver(RiverPollutionProblem, Scipyde_method)
    ideal, nadir = solver.solve()
    expected_ideal = np.array([-6.34, -3.44, -7.50, 0.0])

    assert np.all(np.isclose(ideal, expected_ideal, rtol=0, atol=0.1))
    assert np.all(np.greater(nadir, ideal))
