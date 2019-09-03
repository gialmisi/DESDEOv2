import numpy as np
import pytest
from pytest import approx

from desdeov2.problem.Objective import ObjectiveError, ScalarObjective


@pytest.fixture
def scalar_obj_1():
    def fun(vec):
        return vec[0] + vec[1] + vec[2]

    return ScalarObjective("Objective 1", fun, -10, 10)


@pytest.fixture
def variables_xyz():
    variables = np.array([5.5, -4.25, 0.05])

    return variables


def test_init(scalar_obj_1):
    assert scalar_obj_1.value == approx(0.0)
    assert scalar_obj_1.name == "Objective 1"
    assert scalar_obj_1.lower_bound == approx(-10.0)
    assert scalar_obj_1.upper_bound == approx(10.0)


def test_evaluator(scalar_obj_1, variables_xyz):
    result = scalar_obj_1.evaluate(variables_xyz)
    assert result == approx(5.5 + (-4.25) + 0.05)
    assert result == approx(scalar_obj_1.value)


def test_bad_evaluate(scalar_obj_1, variables_xyz):
    with pytest.raises(ObjectiveError):
        scalar_obj_1.evaluate(variables_xyz[1:])


def test_default_bounds():
    obj = ScalarObjective("", None)
    assert obj.lower_bound == -np.inf
    assert obj.upper_bound == np.inf


def test_bad_bounds():
    with pytest.raises(ObjectiveError):
        ScalarObjective("", None, -10.0, -11.0)
