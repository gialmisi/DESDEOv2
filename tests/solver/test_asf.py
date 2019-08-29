import numpy as np
import pytest
from pytest import approx

from desdeo.solver.ASF import (
    ASFError,
    ReferencePointASF,
    SimpleASF,
    MaxOfTwoASF,
    StomASF,
    PointMethodASF,
    AugmentedGuessASF,
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
    assert np.all(np.isclose(asf.nadir, nadir))
    assert np.all(np.isclose(asf.utopian_point, utopian))

    assert asf.rho == approx(0.1)

    asf.rho = 5
    assert asf.rho == approx(5)


def test_reference_point_call():
    pref_facs = np.array([0.25, 0.5, 0.33, 0.4])
    nadir = np.array([50, 20, 30, 50])
    utopian = np.array([1.5, 8, 3, 7.2])
    rho = 0.2

    asf = ReferencePointASF(pref_facs, nadir, utopian, rho=rho)
    objective = np.array([10, 12, 15, 22])
    reference = np.array([22, 13, 20, 33])

    max_term = np.max(pref_facs * (objective - reference))
    sum_term = rho * np.sum((objective - reference) / (nadir - utopian))
    expected = max_term + sum_term

    res = asf(objective, reference)

    assert res == approx(expected)


def test_maxoftwo(four_dimenional_data_with_extremas):
    _, fs, nadir, ideal = four_dimenional_data_with_extremas
    reference = np.array([5.2, 0.5, 8.5, 2.3])
    asf = MaxOfTwoASF(nadir, ideal, [2, 3], [1])

    res = asf(fs, reference)
    assert np.all(
        np.isclose(res, [1.07353, 0.32653076, 0.93877678, 0.91176686])
    )

    res_single = asf(fs[1], reference)
    assert np.isclose(res_single, 0.32653076)


def test_stom(four_dimenional_data_with_extremas):
    _, fs, _, ideal = four_dimenional_data_with_extremas
    reference = np.array([5.2, 0.5, 8.5, 2.3])
    asf = StomASF(ideal)

    res = asf(fs, reference)
    assert np.all(
        np.isclose(res, [1.00000067, 1.52381026, 4.38095575, 1.76923358])
    )

    res_single = asf(fs[3], reference)
    assert np.isclose(res_single, 1.76923358)


def test_point_method(four_dimenional_data_with_extremas):
    _, fs, nadir, ideal = four_dimenional_data_with_extremas
    reference = np.array([5.2, 0.5, 8.5, 2.3])
    asf = PointMethodASF(nadir, ideal)

    res = asf(fs, reference)
    assert np.all(
        np.isclose(
            res,
            [7.53767183e-07, 1.12245068e-01, 7.24491094e-01, 4.02012319e-01],
        )
    )

    res_single = asf(fs[2], reference)
    assert np.isclose(res_single, 7.24491094e-01)


def test_augmented_guess(four_dimenional_data_with_extremas):
    xs, fs, nadir, ideal = four_dimenional_data_with_extremas
    reference = np.array([5.2, 0.5, 8.5, 2.3])
    asf = AugmentedGuessASF(nadir, ideal, [1, 3])

    res = asf(fs, reference)
    assert np.all(
        np.isclose(res, [-1.00001815, 11.79999576, 13.59999945,  1.1999892])
    )

    res_single = asf(fs[0], reference)
    assert np.isclose(res_single, -1.00001815)
