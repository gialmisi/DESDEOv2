import pytest

from desdeo.methods.Nimbus import SNimbus
from desdeo.methods.InteractiveMethod import InteractiveMethodError
from desdeo.problem.Problem import ScalarDataProblem


import numpy as np


@pytest.fixture
def sphere_nimbus(sphere_pareto):
    problem = ScalarDataProblem(*sphere_pareto)
    method = SNimbus(problem)
    return method


def test_no_pareto_given(sphere_nimbus):
    with pytest.raises(InteractiveMethodError):
        method_bad = SNimbus()
        method_bad.initialize(5)


def test_bad_start_point(sphere_nimbus):
    with pytest.raises(InteractiveMethodError):
        sphere_nimbus.initialize(5, starting_point=np.array([1, 2]))


def test_bad_n_of_points(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(2)

    with pytest.raises(InteractiveMethodError):
        method.initialize(5)

    with pytest.raises(InteractiveMethodError):
        method.initialize(-2)


def test_starting_point(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(3, starting_point=np.array([1.2, 3.2, 4.4]))

    assert np.all(np.isclose(method.current_point, np.array([1.2, 3.2, 4.4])))

    method.initialize(3)

    assert not np.all(
        np.isclose(method.current_point, np.array([1.2, 3.2, 4.4]))
    )


def test_nadir_and_ideal(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(3)

    assert np.all(np.isclose(method.nadir, [1, 1, 1]))
    assert np.all(np.isclose(method.ideal, [0, 0, 0]))


def test_first_iteration(sphere_nimbus):
    method = sphere_nimbus
    res_init = method.initialize(3)

    assert method.first_iteration
    res_iter = method.iterate()

    assert np.all(np.isclose(res_init, res_iter))
    assert not method.first_iteration


def test_bad_classificaitons(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(3, starting_point=np.array([2, 4, 3]))
    method.iterate()

    # wrong number of classificaiton
    with pytest.raises(InteractiveMethodError):
        method.interact([("<", 0), ("=", 0)])

    # wrong type of classification
    with pytest.raises(InteractiveMethodError):
        method.interact([("<", 0), ("=", 0), ("==", 0)])

    # wrong type of paramter for <=
    with pytest.raises(InteractiveMethodError):
        method.interact([("<=", 3), ("=", 0), ("=", 0)])

    # wrong type of paramter for >=
    with pytest.raises(InteractiveMethodError):
        method.interact([("<=", 1), ("=", 0), (">=", 2)])

    # bad classificaton
    with pytest.raises(InteractiveMethodError):
        method.interact([("<=", 1), ("<", 0), ("<=", 2)])


def test_sort_classificaitons(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(4, starting_point=np.array([3.2, 4.1, 5.5]))
    method.iterate()

    method.classifications = [("<", 0), ("<=", 3.5), ("0", 0)]
    method._sort_classsifications()

    assert method._SNimbus__ind_set_lt[0] == 0
    assert method._SNimbus__ind_set_lte[0] == 1
    assert method._SNimbus__ind_set_free[0] == 2

    assert method._SNimbus__aspiration_levels[0] == (1, 3.5)

    method.classifications = [("<", 0), (">=", 5.0), ("=", 0)]
    method._sort_classsifications()

    assert method._SNimbus__ind_set_lt[0] == 0
    assert method._SNimbus__ind_set_gte[0] == 1
    assert method._SNimbus__ind_set_eq[0] == 2

    assert method._SNimbus__upper_bounds[0] == (1, 5.0)


@pytest.mark.snipe
def test_iterate(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(4, starting_point=np.array([3.2, 4.1, 5.5]))

    method.iterate()
    method.interact([("<", 0), (">=", 5.0), ("=", 0)])
    method.iterate()
