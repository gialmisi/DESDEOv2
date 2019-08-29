import numpy as np
import pytest

from desdeo.problem.Problem import ScalarDataProblem


@pytest.fixture
def dummy_data():
    xs = np.array([[1, 3, 4], [2, 2, 3], [3, 2, 1]])
    fs = np.array([[10, 12], [13, 22], [11, 19]])
    return xs, fs


@pytest.fixture
def dummy_problem(dummy_data):
    xs, fs = dummy_data
    return ScalarDataProblem(xs, fs)


def test_init(dummy_problem, dummy_data):
    xs, fs = dummy_data
    problem = dummy_problem

    assert np.all(np.isclose(xs, problem.decision_vectors))
    assert np.all(np.isclose(fs, problem.objective_vectors))

    assert 3 == problem.n_of_variables
    assert 2 == problem.n_of_objectives

    assert np.all(np.isclose(problem.nadir, [13, 22]))
    assert np.all(np.isclose(problem.ideal, [10, 12]))


def test_get_variable_bounds(dummy_problem):
    problem = dummy_problem

    bounds = problem.get_variable_bounds()
    assert np.all(np.isclose(bounds[0], [1, 3]))
    assert np.all(np.isclose(bounds[1], [2, 3]))
    assert np.all(np.isclose(bounds[2], [1, 4]))


def test_evaluate(dummy_problem, dummy_data):
    xs, fs = dummy_data
    problem = dummy_problem
    origin = np.array([0, 0, 0])

    assert np.all(np.isclose(problem.evaluate(xs[0]), fs[0]))
    assert np.all(np.isclose(problem.evaluate(xs[1]), fs[1]))
    assert np.all(np.isclose(problem.evaluate(xs[2]), fs[2]))

    assert np.all(np.isclose(problem.evaluate(origin), fs[2]))
