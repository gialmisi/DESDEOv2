import numpy as np
import pytest
from pytest import approx

from desdeo.solver.ASF import (
    ASFError,
    ReferencePointASF,
    SimpleASF,
    MaxOfTwoASF,
)


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


def test_reference_point_init():
    pref_facs = np.array([0.25, 0.5, 0.33, 0.4])
    nadir = np.array([50, 20, 30, 50])
    utopian = np.array([1.5, 8, 3, 7.2])

    asf = ReferencePointASF(pref_facs, nadir, utopian)

    assert np.all(np.isclose(asf.preferential_factors, pref_facs))
    assert np.all(np.isclose(asf.nadir_point, nadir))
    assert np.all(np.isclose(asf.utopian_point, utopian))

    assert asf.roo == approx(0.1)

    asf.roo = 5
    assert asf.roo == approx(5)


def test_reference_point_call():
    pref_facs = np.array([0.25, 0.5, 0.33, 0.4])
    nadir = np.array([50, 20, 30, 50])
    utopian = np.array([1.5, 8, 3, 7.2])
    roo = 0.2

    asf = ReferencePointASF(pref_facs, nadir, utopian, roo=roo)
    objective = np.array([10, 12, 15, 22])
    reference = np.array([22, 13, 20, 33])

    max_term = np.max(pref_facs * (objective - reference))
    sum_term = roo * np.sum((objective - reference) / (nadir - utopian))
    expected = max_term + sum_term

    res = asf(objective, reference)

    assert res == approx(expected)


@pytest.mark.snipe
def test_maxoftwo(simple_data):
    nadir = np.array([50, 20])
    ideal = np.array([-20, -15])
    reference = np.array([5, 2])
    rho = 0.2
    asf = MaxOfTwoASF(nadir, ideal, [0, 1], [], rho=rho)
    xs, fs = simple_data

    res = asf(fs, reference)
    print(res)
