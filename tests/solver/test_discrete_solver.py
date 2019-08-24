import numpy as np
import pytest
from pytest import approx

from desdeo.solver.ASF import SimpleASF
from desdeo.solver.NumericalMethods import DiscreteMinimizer
from desdeo.solver.ScalarSolver import (
    ASFScalarSolver,
    EpsilonConstraintScalarSolver,
    ScalarSolverError,
    WeightingMethodScalarSolver,
)
from desdeo.problem.Problem import ScalarConstraint


def test_weighting(simple_data_problem):
    solver = WeightingMethodScalarSolver(
        simple_data_problem, DiscreteMinimizer()
    )
    res1 = solver.solve(np.array([1, 1]))

    assert np.all(np.isclose(res1[0], [-1.05, -2.05, 3.1]))
    assert np.all(np.isclose(res1[1], [0, 3.861994]))

    res2 = solver.solve(np.array([0, 0.5]))

    assert np.all(np.isclose(res2[0], [2, 2, 2]))
    assert np.all(np.isclose(res2[1], [6, 3.464101]))


def test_epsilon(simple_data_problem):
    solver = EpsilonConstraintScalarSolver(
        simple_data_problem, DiscreteMinimizer()
    )
    solver.epsilons = np.array([10.0, 5.0])
    res1 = solver.solve(0)

    assert np.all(np.isclose(res1[0], [-1.05, -2.05, 3.1]))
    assert np.all(np.isclose(res1[1], [0, 3.861994]))

    solver.epsilons = np.array([5.0, 10.0])
    res2 = solver.solve(1)

    assert np.all(np.isclose(res2[0], [-1.05, -2.05, 3.1]))
    assert np.all(np.isclose(res2[1], [0, 3.861994]))

    solver.epsilons = np.array([20, 20])
    res3 = solver.solve(1)

    assert np.all(np.isclose(res3[0], [2, 2, 2]))
    assert np.all(np.isclose(res3[1], [6, 3.464101]))


def test_asf(simple_data_problem):
    solver = ASFScalarSolver(simple_data_problem, DiscreteMinimizer())

    solver.asf = SimpleASF([1, 1])
    res1 = solver.solve(np.array([6, 3.4]))

    assert np.all(np.isclose(res1[0], [2, 2, 2]))
    assert np.all(np.isclose(res1[1], [6, 3.464101]))

    res2 = solver.solve(np.array([0, 0]))

    assert np.all(np.isclose(res2[0], [-1.05, -2.05, 3.1]))
    assert np.all(np.isclose(res2[1], [0, 3.861994]))


def test_asf_with_cons(simple_data_problem):
    solver = ASFScalarSolver(simple_data_problem, DiscreteMinimizer())

    def fun1(xs, fs):
        return fs[:, 0] - 7

    cons1 = ScalarConstraint("cons1", 3, 2, fun1)
    simple_data_problem.constraints = [cons1]

    solver.asf = SimpleASF([1, 1])
    res1 = solver.solve(np.array([6, 3.4]))

    assert np.all(np.isclose(res1[0], [2.2, 3.3, 6.6]))
    assert np.all(np.isclose(res1[1], [12.1, 7.699999]))
