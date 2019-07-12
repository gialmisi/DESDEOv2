import numpy as np
from pytest import approx

from desdeo.solver.PointSolver import IdealAndNadirPointSolver


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


def test_ideal_and_nadir_point_solver(CylinderProblem):
    solver = IdealAndNadirPointSolver(CylinderProblem)
    ideal, nadir = solver.solve()
    expected_ideal = np.array([785.476703213788, -2945.321652556771, 0.0])

    assert np.all(np.isclose(ideal, expected_ideal, rtol=0, atol=0.5))
    assert np.all(np.greater(nadir, ideal))


def test_ideal_and_nadir_point_solver_river(RiverPollutionProblem):
    solver = IdealAndNadirPointSolver(RiverPollutionProblem)
    ideal, nadir = solver.solve()
    expected_ideal = np.array([-6.34, -3.44, -7.50, 0.0])

    assert np.all(np.isclose(ideal, expected_ideal, rtol=0, atol=0.1))
    assert np.all(np.greater(nadir, ideal))
