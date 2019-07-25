from desdeo.methods.EvolutionaryMethod import MOEAD, EvolutionaryError

import numpy as np

import pytest


def test_initialize(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = MOEAD(problem)

    weights = np.ones((10, problem.n_of_objectives))
    method.initialize(10, 3, weights)

    assert method.n == 10
    assert method.t == 3
    assert method.lambdas.shape == (10, problem.n_of_objectives)
    assert np.all(np.isclose(method.lambdas, weights))

    with pytest.raises(EvolutionaryError):
        # bad t
        method.t = 20

        # too many weights
        method.lambdas = np.ones((100, problem.n_of_objectives))

        # weight vectors too short or too long
        method.lambdas = np.ones((method.n, 10))
        method.lambdas = np.ones((method.n, 2))


@pytest.mark.snipe
def test_generate_uniform_weights(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = MOEAD(problem)

    dummy = np.ones((10, problem.n_of_objectives))
    method.initialize(10, 3, dummy)

    res = method._generate_uniform_set_of_weights()
    print(res)
