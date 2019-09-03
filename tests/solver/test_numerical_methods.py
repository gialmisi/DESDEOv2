import numpy as np
import pytest

from desdeov2.solver.NumericalMethods import (
    DiscreteMinimizer,
    NumericalMethodError,
)


@pytest.fixture
def minimizer_method():
    fun = DiscreteMinimizer.minimizer
    return fun


@pytest.fixture
def simple_data():
    xs = np.array([[1, 2], [2.1, 3], [3, 7], [1.5, 3]])

    fs = np.array([[-1, -3, -2], [3, 5, 3], [-1, -1, -2], [1, 1, 1]])

    bounds = np.array([[0.5, 3], [2.5, 10]])

    return xs, fs, bounds


@pytest.fixture
def evaluator():
    evaluator = lambda x, y: np.sum(y, axis=1)  # noqa
    return evaluator


def test_minimizer_method(minimizer_method, simple_data, evaluator):
    fun = minimizer_method
    xs, fs, bounds = simple_data

    res = fun(None, evaluator, bounds, xs, fs)
    assert np.all(np.isclose(res, [1.5, 3]))


def test_minimizer_missing_parameters(
    minimizer_method, simple_data, evaluator
):
    fun = minimizer_method
    xs, fs, bounds = simple_data

    # no var or objs
    with pytest.raises(NumericalMethodError):
        fun(None, evaluator, bounds)

    # var no objs
    with pytest.raises(NumericalMethodError):
        fun(None, evaluator, bounds, variables=xs)

    # no var objs
    with pytest.raises(NumericalMethodError):
        fun(None, evaluator, bounds, objectives=fs)

    # None bounds, or empty
    res = fun(None, evaluator, None, xs, fs)
    assert np.array_equiv(res, [1, 2])


def test_discrete_minimizer_numerical_method(simple_data, evaluator):
    xs, fs, bounds = simple_data
    method = DiscreteMinimizer()

    res1 = method.run(evaluator, bounds=None, variables=xs, objectives=fs)
    res2 = method.run(evaluator, bounds, variables=xs, objectives=fs)

    assert np.array_equiv(res1, [1, 2])
    assert np.array_equiv(res2, [1.5, 3])
