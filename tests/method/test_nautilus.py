import numpy as np
import pytest

from desdeo.methods.Nautilus import InteractiveMethodError, Nautilus


def test_nautilus_initialization(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = Nautilus(problem)
    # dont compute ideal and nadir
    method.problem.nadir = 1
    method.problem.ideal = 1

    init_reqs = method.initialization_requirements[0]

    # Test that the caller works
    assert method.n_of_iterations == 0
    init_reqs[2](method, 5)
    assert method.n_of_iterations == 5

    # Test initializer
    init_pars = {init_reqs[0]: 20}
    method.initialize(init_pars)
    assert method.n_of_iterations == 20

    # Test bad init pars
    init_pars_bad_key = {"Wrong key": 10}
    with pytest.raises(InteractiveMethodError):
        method.initialize(init_pars_bad_key)

    # Test wrong type
    init_pars_wrong_type = {init_reqs[0]: "14"}
    with pytest.raises(InteractiveMethodError):
        method.initialize(init_pars_wrong_type)


def test_nautilus_initialization_ideal_and_nadir(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = Nautilus(problem)
    expected_ideal = np.array([-6.34, -3.44, -7.50, 0.0])

    assert method.problem.ideal is None
    assert method.problem.nadir is None

    # Both missing
    method.initialize({"Number of iterations": 5})

    assert np.all(
        np.isclose(method.problem.ideal, expected_ideal, rtol=0, atol=0.1)
    )
    assert np.all(np.greater(method.problem.nadir, method.problem.ideal))

    # Ideal missing
    method.problem.ideal = None
    assert method.problem.ideal is None
    method.initialize({"Number of iterations": 5})

    assert np.all(
        np.isclose(method.problem.ideal, expected_ideal, rtol=0, atol=0.1)
    )
    assert np.all(np.greater(method.problem.nadir, method.problem.ideal))

    # Nadir missing
    method.problem.nadir = None
    assert method.problem.nadir is None
    method.initialize({"Number of iterations": 5})

    assert np.all(
        np.isclose(method.problem.ideal, expected_ideal, rtol=0, atol=0.1)
    )
    assert np.all(np.greater(method.problem.nadir, method.problem.ideal))
