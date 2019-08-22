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
