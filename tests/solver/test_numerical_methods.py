import pytest

import numpy as np

from desdeo.solver.NumericalMethods import (
    DiscreteMinimizer,
    NumericalMethodError,
    )


@pytest.fixture
def minimizer_method():
    fun = DiscreteMinimizer.minimizer
    return fun


@pytest.fixture
def simple_data():
    xs = np.array([
        [1, 2],
        [2.1, 3],
        [3, 7],
        [1.5, 3],
    ])

    fs = np.array([
        [-1, -3, -2],
        [3, 5, 3],
        [-1, -1, -2],
        [1, 1, 1]
    ])

    bounds = np.array([
        [0.5, 3],
        [2.5, 10]
    ])

    return xs, fs, bounds


def test_minimizer_method(minimizer_method, simple_data):
    fun = minimizer_method
    xs, fs, bounds = simple_data

    evaluator = lambda x: np.sum(x, axis=1)  # noqa

    res = fun(evaluator, bounds, xs, fs)
    assert np.all(np.isclose(res, [1.5, 3]))


def test_minimizer_no_data(minimizer_method, simple_data):
    fun = minimizer_method
    xs, fs, bounds = simple_data
    evaluator = lambda x: np.sum(x, axis=1)  # noqa

    # no var or objs
    with pytest.raises(NumericalMethodError):
        fun(evaluator, bounds)

    # var no objs
    with pytest.raises(NumericalMethodError):
        fun(evaluator, bounds, variables=xs)

    # no var objs
    with pytest.raises(NumericalMethodError):
        fun(evaluator, bounds, objectives=fs)
