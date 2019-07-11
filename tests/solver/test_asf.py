import numpy as np
import pytest
from pytest import approx

from desdeo.solver.ASF import ASFError, SimpleASF


def test_simple_init():
    weights_1 = np.array([1, 2, 3])
    simple_asf = SimpleASF(weights_1)
    weights_tmp = weights_1
    weights_1 = np.array([1, 2, 4])

    assert np.all(np.isclose(simple_asf.weights, weights_tmp))
    assert not np.all(np.isclose(simple_asf.weights, weights_1))


def test_simple_setter():
    weights_1 = np.array([1, 2, 3])
    simple_asf = SimpleASF(weights_1)
    weights_2 = np.array([1, 2, 4])

    assert np.all(np.isclose(simple_asf.weights, weights_1))
    assert not np.all(np.isclose(simple_asf.weights, weights_2))

    simple_asf.weights = weights_2

    assert np.all(np.isclose(simple_asf.weights, weights_2))
    assert not np.all(np.isclose(simple_asf.weights, weights_1))


def test_simple_call():
    weights = np.array([3, 4, 2])
    simple_asf = SimpleASF(weights)

    objective = np.array([1, 1, 2.5])
    reference = np.array([0.5, -2, 1.5])

    res = simple_asf(objective, reference)

    assert res == approx(12.0)


def test_simple_non_matching_shapes():
    weights = np.ones(3)
    simple_asf = SimpleASF(weights)

    objective = np.array([1, 1, 2.5])
    reference = np.array([0.5, -2, 1.5])

    objective2d = np.array([[1, 1, 1], [2, 2, 2]])
    reference2d = np.array([[1, 1, 1], [2, 2, 2]])

    with pytest.raises(ASFError):
        simple_asf(objective[:1], reference)

    with pytest.raises(ASFError):
        simple_asf(objective2d, reference)

    with pytest.raises(ASFError):
        simple_asf(objective, reference2d)
