import numpy as np
import pytest
from pytest import approx

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
    method.initialize(initialization_parameters=init_pars)
    assert method.n_of_iterations == 20

    # Test bad init pars
    init_pars_bad_key = {"Wrong key": 10}
    with pytest.raises(InteractiveMethodError):
        method.initialize(initialization_parameters=init_pars_bad_key)

    # Test wrong type
    init_pars_wrong_type = {init_reqs[0]: "14"}
    with pytest.raises(InteractiveMethodError):
        method.initialize(initialization_parameters=init_pars_wrong_type)

    # Test negatigve
    init_pars_neg_iters = {init_reqs[0]: -1}
    with pytest.raises(InteractiveMethodError):
        method.initialize(initialization_parameters=init_pars_neg_iters)

    # Test direct specification
    itn = 42
    method.initialize(itn=itn)
    assert method.n_of_iterations == 42


def test_nautilus_initialization_ideal_and_nadir(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = Nautilus(problem)
    expected_ideal = np.array([-6.34, -3.44, -7.50, 0.0])

    assert method.problem.ideal is None
    assert method.problem.nadir is None

    # Both missing
    method.initialize(initialization_parameters={"Number of iterations": 5})

    assert np.all(
        np.isclose(method.problem.ideal, expected_ideal, rtol=0, atol=0.1)
    )
    assert np.all(np.greater(method.problem.nadir, method.problem.ideal))

    # Ideal missing
    method.problem.ideal = None
    assert method.problem.ideal is None
    method.initialize(initialization_parameters={"Number of iterations": 5})

    assert np.all(
        np.isclose(method.problem.ideal, expected_ideal, rtol=0, atol=0.1)
    )
    assert np.all(np.greater(method.problem.nadir, method.problem.ideal))

    # Nadir missing
    method.problem.nadir = None
    assert method.problem.nadir is None
    method.initialize(initialization_parameters={"Number of iterations": 5})

    assert np.all(
        np.isclose(method.problem.ideal, expected_ideal, rtol=0, atol=0.1)
    )
    assert np.all(np.greater(method.problem.nadir, method.problem.ideal))


def test_nautilus_iterations_setter(RiverPollutionProblem):
    method = Nautilus(RiverPollutionProblem)

    method.n_of_iterations = 10
    assert method.n_of_iterations == 10

    with pytest.raises(InteractiveMethodError):
        method.n_of_iterations = -5


def test_nautilus_index_setter(RiverPollutionProblem):
    method = Nautilus(RiverPollutionProblem)

    method.preference_index_set = np.array([1, 2, 2, 3])
    assert np.all(method.preference_index_set == np.array([1, 2, 2, 3]))

    with pytest.raises(InteractiveMethodError):
        method.preference_index_set = np.array([1, 2])


def test_nautilus_percentage_setter(RiverPollutionProblem):
    method = Nautilus(RiverPollutionProblem)

    method.preference_percentages = np.array([25, 10, 10, 55])
    assert np.all(
        np.isclose(method.preference_percentages, np.array([25, 10, 10, 55]))
    )

    # wrong length
    with pytest.raises(InteractiveMethodError):
        method.preference_percentages = np.array([25, 25, 50])

    # bad sum
    with pytest.raises(InteractiveMethodError):
        method.preference_percentages = np.array([25, 20, 50, 25])


def test_nautilus_iterate_preference(NautilusInitializedRiver):
    method = NautilusInitializedRiver
    method.problem.ideal = np.array([0.2, 0.3, 0.1, 0.5])
    method.problem.nadir = np.array([20, 30, 10, 50])

    # test dict, relative
    pref_dict_rel = {"Relative importance": np.array([1, 1, 2, 3])}
    method.iterate(preference_information=pref_dict_rel)
    assert np.all(method.preference_index_set == np.array([1, 1, 2, 3]))

    # test dict, percentage
    pref_dict_per = {"Percentages": np.array([25, 25, 25, 25])}
    method.iterate(preference_information=pref_dict_per)
    assert np.all(
        np.isclose(method.preference_percentages, np.array([25, 25, 25, 25]))
    )

    # test named parameter, relative
    method.iterate(index_set=np.array([1, 2, 3, 3]))
    assert np.all(method.preference_index_set == np.array([1, 2, 3, 3]))

    # test named paramter, percentage
    method.iterate(percentages=np.array([10, 20, 40, 30]))
    assert np.all(
        np.isclose(method.preference_percentages, np.array([10, 20, 40, 30]))
    )

    # test named, default
    method.iterate()
    assert np.all(
        np.isclose(method.preference_percentages, np.array([25, 25, 25, 25]))
    )

    # test bad
    with pytest.raises(InteractiveMethodError):
        dict_bad = {"Bad value": "hello"}
        method.iterate(preference_information=dict_bad)


def test_nautilus_iterate_index_set(RiverPollutionProblem):
    problem = RiverPollutionProblem
    problem.ideal = np.array([0.2, 0.3, 0.1, 0.5])
    problem.nadir = np.array([20, 30, 10, 50])
    method = Nautilus(problem)
    method.epsilon = 0.1

    method.initialize()
    method.iterate(index_set=np.array([1, 2, 2, 4]))

    mu = method.preferential_factors

    assert mu[0] == approx(1 / (1 * (20 - (0.2 - 0.1))))
    assert mu[1] == approx(1 / (2 * (30 - (0.3 - 0.1))))
    assert mu[2] == approx(1 / (2 * (10 - (0.1 - 0.1))))
    assert mu[3] == approx(1 / (4 * (50 - (0.5 - 0.1))))


def test_nautilus_iterate_percentages(RiverPollutionProblem):
    problem = RiverPollutionProblem
    problem.ideal = np.array([0.2, 0.3, 0.1, 0.5])
    problem.nadir = np.array([20, 30, 10, 50])
    method = Nautilus(problem)
    method.epsilon = 0.1

    method.initialize()
    method.iterate(percentages=np.array([20, 25, 15, 40]))

    mu = method.preferential_factors

    assert mu[0] == approx(1 / ((20 / 100) * (20 - (0.2 - 0.1))))
    assert mu[1] == approx(1 / ((25 / 100) * (30 - (0.3 - 0.1))))
    assert mu[2] == approx(1 / ((15 / 100) * (10 - (0.1 - 0.1))))
    assert mu[3] == approx(1 / ((40 / 100) * (50 - (0.5 - 0.1))))


def test_nautilus_iterate_first_point(NautilusInitializedRiver):
    method = NautilusInitializedRiver
    method.iterate(index_set=np.array([2, 2, 1, 1]))

    objective = method._Nautilus__objective_vectors[1]
    solution = method._Nautilus__solutions[1]

    evaluated = method.problem.evaluate(solution)[0][0]

    assert np.all(np.isclose(objective, evaluated))
