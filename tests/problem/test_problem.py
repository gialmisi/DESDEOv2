import numpy as np
import pytest
from pytest import approx

from desdeo.problem.Constraint import ScalarConstraint
from desdeo.problem.Objective import ScalarObjective
from desdeo.problem.Problem import ProblemError, ScalarMOProblem
from desdeo.problem.Variable import Variable


@pytest.fixture
def scalar_objectives():
    def fun1(d_vars):
        return d_vars[0] + d_vars[1] + d_vars[2]

    def fun2(d_vars):
        return d_vars[3] - d_vars[1] * d_vars[2]

    def fun3(d_vars):
        return d_vars[2]

    obj_1 = ScalarObjective("f1", fun1)
    obj_2 = ScalarObjective("f2", fun2)
    obj_3 = ScalarObjective("f3", fun3)

    return [obj_1, obj_2, obj_3]


@pytest.fixture
def scalar_constraints():
    def cons1(d_vars, obj_funs):
        # GT
        gt = 5.0

        res = gt - d_vars[0] + obj_funs[2]
        return res

    def cons2(d_vars, obj_funs):
        # LT
        lt = 15.5
        res = obj_funs[0] * obj_funs[1] - d_vars[2] - lt
        return res

    def cons3(d_vars, obj_funs):
        # EQ
        eq = -2.5
        res = d_vars[3] - eq

        return -abs(res)

    cons_1 = ScalarConstraint("cons_1", 4, 3, cons1)
    cons_2 = ScalarConstraint("cons_2", 4, 3, cons2)
    cons_3 = ScalarConstraint("cons_3", 4, 3, cons3)

    return [cons_1, cons_2, cons_3]


@pytest.fixture
def variables():
    var_1 = Variable("x", 2.5, 0.0, 10.0)
    var_2 = Variable("y", -5.8, -12.0, -5.5)
    var_3 = Variable("z", -10.5, -20, 5.5)
    var_4 = Variable("a", 14.2, 10.2, 42.5)

    return [var_1, var_2, var_3, var_4]


@pytest.fixture
def simple_scalar_moproblem(scalar_objectives, variables, scalar_constraints):
    return ScalarMOProblem(scalar_objectives, variables, scalar_constraints)


@pytest.fixture
def nice_population():
    """A well behaved population, no bounds are broken.

    """
    return np.array(
        [
            [2.0, -9.2, 2.2, 30.8],
            [9.8, -11.1, -1.5, 15.5],
            [0.02, -5.4, 4.2, 10.3],
            [2.5, -5.8, -10.5, 14.2],
        ]
    )


def test_init(simple_scalar_moproblem):
    p = simple_scalar_moproblem

    assert p.n_of_objectives == 3
    assert p.n_of_variables == 4
    assert p.n_of_constraints == 3
    assert p.nadir is None
    assert p.ideal is None


def test_evaluate(simple_scalar_moproblem, nice_population):
    # Test the evaluation of a simple problem
    p = simple_scalar_moproblem
    pop = nice_population

    obj_vals, cons_vals = p.evaluate(pop)

    # Check that the evaluated values and constraints are correct
    # 1st row
    assert np.all(np.isclose(np.array([-4.999999, 51.04, 2.2]), obj_vals[0]))
    assert np.all(
        np.isclose(np.array([5.2, -272.89994896, -33.3]), cons_vals[0])
    )

    # 2nd row
    assert np.all(
        np.isclose(np.array([-2.799999, -1.149999, -1.5]), obj_vals[1])
    )
    assert np.all(
        np.isclose(np.array([-6.3, -10.780003949999, -18.0]), cons_vals[1])
    )

    # 3rd row
    assert np.all(np.isclose(np.array([-1.18, 32.98, 4.2]), obj_vals[2]))
    assert np.all(np.isclose(np.array([9.18, -58.6164, -12.8]), cons_vals[2]))

    # 4th row
    assert np.all(np.isclose(np.array([-13.8, -46.7, -10.5]), obj_vals[3]))
    assert np.all(np.isclose(np.array([-8.0, 639.46, -16.7]), cons_vals[3]))


def test_evaluate_wrong_length(simple_scalar_moproblem):
    p = simple_scalar_moproblem
    bad_pop = np.array([[1, 2, 3], [3, 3, 3], [1, 2, 3]])

    with pytest.raises(ProblemError):
        p.evaluate(bad_pop)


def test_bad_nadir(scalar_objectives, variables, scalar_constraints):
    with pytest.raises(ProblemError):
        ScalarMOProblem(
            scalar_objectives,
            variables,
            scalar_constraints,
            nadir=[1, 4],
            ideal=None,
        )


def test_bad_ideal(scalar_objectives, variables, scalar_constraints):
    with pytest.raises(ProblemError):
        ScalarMOProblem(
            scalar_objectives,
            variables,
            scalar_constraints,
            nadir=None,
            ideal=[1, 2],
        )


def test_get_variable_bounds(simple_scalar_moproblem, variables):
    bounds = simple_scalar_moproblem.get_variable_bounds()

    for (fst, snd) in zip(bounds, [var.get_bounds() for var in variables]):
        assert fst == tuple(map(approx, snd))
