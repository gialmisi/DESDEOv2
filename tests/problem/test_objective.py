import pytest
from pytest import approx


from desdeo.problem.Objective import ScalarObjective, ObjectiveError
from desdeo.problem.Variable import Variable


@pytest.fixture
def scalar_obj_1():
    def fun(x, y, z):
        return x + y + z

    return ScalarObjective("Objective 1", fun, ["x", "y", "z"])


@pytest.fixture
def variables_xyz():
    variables = []
    variables.append(Variable("x", 5.5))
    variables.append(Variable("y", -4.25))
    variables.append(Variable("z", 0.05))

    return variables


def test_init(scalar_obj_1):
    assert(scalar_obj_1.value == approx(0.0))
    assert(scalar_obj_1.name == "Objective 1")


def test_evaluator(scalar_obj_1, variables_xyz):
    result = scalar_obj_1.evaluate(variables_xyz)
    assert(result == approx(5.5 + (-4.25) + 0.05))
    assert(result == approx(scalar_obj_1.value))


def test_bad_evaluate(scalar_obj_1, variables_xyz):
    with pytest.raises(ObjectiveError):
        scalar_obj_1.evaluate(variables_xyz[1:])
