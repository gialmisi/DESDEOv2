import numpy as np
import pytest
from pytest import approx

from desdeo.problem.Variable import Variable, VariableError


@pytest.fixture
def x_variable():
    return Variable("x", 5.2, 0.5, 12.3)


@pytest.fixture
def default_variable():
    return Variable("x", 4.3)


def test_get_name(x_variable):
    assert x_variable.name == "x"


def test_get_initial_value(x_variable):
    assert x_variable.initial_value == approx(5.2)


def test_get_bounds(x_variable):
    bounds = x_variable.get_bounds()
    res = [a == approx(b) for (a, b) in zip([0.5, 12.3], bounds)]
    assert all(res)


def test_init_current_value(x_variable):
    assert x_variable.current_value == approx(5.2)


def test_update_current_value(x_variable):
    x_variable.current_value = 78.45
    assert x_variable.current_value == approx(78.45)


def test_bad_bounds():
    with pytest.raises(VariableError):
        Variable("x", 5.3, 2.3, 1.4)


def test_bad_initial_value():
    with pytest.raises(VariableError):
        Variable("x", -42.3, 2.3, 10.2)


def test_default_bounds(default_variable):
    bounds = default_variable.get_bounds()
    res = [a == approx(b) for (a, b) in zip([-np.inf, np.inf], bounds)]
    assert all(res)
