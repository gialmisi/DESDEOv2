import numpy as np
import pytest
from pytest import approx

from desdeov2.problem.Constraint import (
    ConstraintError,
    ScalarConstraint,
    constraint_function_factory,
)


@pytest.fixture
def objective_vector_1():
    return np.array([1.0, 5.0, 10.0])


@pytest.fixture
def decision_vector_1():
    return np.array([-1.0, 10.0, 5.0, -3.5])


@pytest.fixture
def equal_constraint():
    def constraint(decision_vector, objective_vector):
        x = decision_vector
        y = objective_vector

        h = 50  # Must equal this

        res = h - (x[1] * y[2] - x[2] * y[1])

        # An equality constraint is true only when res equals zero
        return -abs(res)

    return constraint


@pytest.fixture
def lt_constraint():
    def constraint(decision_vector, objective_vector):
        x = decision_vector
        y = objective_vector

        lt = 13.5  # Must be less than this

        res = lt - (x[0] * x[2] + y[2] + y[1])

        return res

    return constraint


@pytest.fixture
def gt_constraint():
    def constraint(decision_vector, objective_vector):
        x = decision_vector
        y = objective_vector

        gt = -5.5  # Must be greater than this

        res = ((y[0] * y[2]) / (x[1] * x[2] * x[3])) - gt

        return res

    return constraint


@pytest.fixture
def simple_constraint_factory(decision_vector_1, objective_vector_1):
    def factory(constraint):
        return ScalarConstraint(
            "test", len(decision_vector_1), len(objective_vector_1), constraint
        )

    return factory


def test_init(decision_vector_1, objective_vector_1, equal_constraint):
    name = "test"
    cons = ScalarConstraint(
        name, len(decision_vector_1), len(objective_vector_1), equal_constraint
    )
    assert cons.name == "test"
    assert cons.n_decision_vars == len(decision_vector_1)
    assert cons.n_objective_funs == len(objective_vector_1)

    res1 = equal_constraint(decision_vector_1, objective_vector_1)
    res2 = cons.evaluator(decision_vector_1, objective_vector_1)

    assert res1 == approx(res2)


def test_equal_cons(
    simple_constraint_factory,
    equal_constraint,
    decision_vector_1,
    objective_vector_1,
):
    cons = simple_constraint_factory(equal_constraint)
    res = cons.evaluate(decision_vector_1, objective_vector_1)

    assert res == approx(-25.0)


def test_gt_cons(
    simple_constraint_factory,
    gt_constraint,
    decision_vector_1,
    objective_vector_1,
):
    cons = simple_constraint_factory(gt_constraint)
    res = cons.evaluate(decision_vector_1, objective_vector_1)

    assert res == approx(5.442857142857143)


def test_lt_cons(
    simple_constraint_factory,
    lt_constraint,
    decision_vector_1,
    objective_vector_1,
):
    cons = simple_constraint_factory(lt_constraint)
    res = cons.evaluate(decision_vector_1, objective_vector_1)

    assert res == approx(3.5)


def test_bad_evaluate_call(
    simple_constraint_factory,
    equal_constraint,
    decision_vector_1,
    objective_vector_1,
):
    cons = simple_constraint_factory(equal_constraint)
    # Too few decision variables
    with pytest.raises(ConstraintError):
        cons.evaluate(decision_vector_1[1:], objective_vector_1)

    # Too few objective function values
    with pytest.raises(ConstraintError):
        cons.evaluate(decision_vector_1, objective_vector_1[1:])

    # Too many decision variables
    with pytest.raises(ConstraintError):
        cons.evaluate(np.ones(10), objective_vector_1)

    # Too many objective function values
    with pytest.raises(ConstraintError):
        cons.evaluate(decision_vector_1, np.ones(10))


def test_constraint_function_factory_equal():
    cons = constraint_function_factory(
        lambda x, y: x[0] + x[1] + y[0] + y[1], 10.0, "=="
    )
    decision_vector = np.array([2.5, 7.5])
    objective_vector = np.array([-7.1, 10.2])

    res = cons(decision_vector, objective_vector)

    assert res == approx(-3.1)


def test_constraint_function_factory_lt():
    cons = constraint_function_factory(
        lambda x, y: x[0] + x[1] + y[0] + y[1], 5.0, "<"
    )
    decision_vector = np.array([2.5, 7.5])
    objective_vector = np.array([-7.1, 10.2])

    res = cons(decision_vector, objective_vector)

    assert res == approx(-8.1)


def test_constraint_function_factory_gt():
    cons = constraint_function_factory(
        lambda x, y: x[0] + x[1] + y[0] + y[1], 9.5, ">"
    )
    decision_vector = np.array([2.5, 7.5])
    objective_vector = np.array([-7.1, 10.2])

    res = cons(decision_vector, objective_vector)

    assert res == approx(3.6)


def test_constraint_function_factory_bad_operator():
    with pytest.raises(ValueError):
        constraint_function_factory(lambda x: x[0] + x[1], 9.5, "x")
