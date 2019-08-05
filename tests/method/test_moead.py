import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt

from desdeo.methods.EvolutionaryMethod import MOEAD, EvolutionaryError


def test_initialize(CylinderProblem):
    problem = CylinderProblem
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

    assert method.b.shape == (method.n, method.t)

    assert method.pop.shape == (method.n, method.problem.n_of_variables)

    assert method.fs.shape == (method.n, method.problem.n_of_objectives)

    assert method.z.shape == (method.problem.n_of_objectives,)


def test_generate_uniform_weights(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = MOEAD(problem)

    dummy = np.ones((10, problem.n_of_objectives))
    method.initialize(10, 3, dummy)

    res = method._generate_uniform_set_of_weights()
    assert np.all(np.logical_and(0 <= res, res < 1.0))


def test_compute_neighborhoods():
    method = MOEAD(None)

    n = 2000
    t = 1000
    n_objectives = 3

    method.n = n
    method.t = t

    # Random weights
    lambdas_rand = np.random.uniform(size=(n, n_objectives))
    method._MOEAD__lambdas = lambdas_rand
    b_rand = method._compute_neighborhoods()

    # The neighborhood of each weight should not contain the weight itself the
    # neighborhood in calculated on
    assert all((
        [i not in neighborhood for (i, neighborhood) in enumerate(b_rand)]
    ))

    # With random weights, we should expect the mean of each neighborhood to be
    # close to n/2 (since we have t random indices in each neighborhood ranging
    # between 0 and n-1). Allow for a 2.5% error relative to n.
    assert np.all(np.isclose(np.mean(b_rand, axis=1), 0.5*n, atol=0.025*n))


def test_generate_feasible_population(CylinderProblem, RiverPollutionProblem):
    # problem with constraints
    method = MOEAD(CylinderProblem)
    method.n = 100

    pop, fs = method._generate_feasible_population()
    feval, cons = method.problem.evaluate(pop)

    assert pop.shape == (method.n, method.problem.n_of_variables)
    assert np.all(cons >= 0)
    assert fs.shape == (method.n, method.problem.n_of_objectives)
    assert np.all(np.isclose(fs, feval))

    # problem with no constraints
    method.problem = RiverPollutionProblem

    pop, fs = method._generate_feasible_population()
    feval, cons = method.problem.evaluate(pop)

    assert pop.shape == (method.n, method.problem.n_of_variables)
    assert np.all(np.equal(cons, None))
    assert fs.shape == (method.n, method.problem.n_of_objectives)
    assert np.all(np.isclose(fs, feval))


@pytest.mark.snipe
def test_run(DTLZ1_3D):
    method = MOEAD(DTLZ1_3D)
    method.initialize(50, 5)
    for i in range(100):
        print(i)
        method.run()

    ep_x = np.array(method.epop)
    ep = np.array(method.epop_fs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2])
    # ax.set_xlim3d(0, 1)
    # ax.set_ylim3d(0, 1)
    # ax.set_zlim3d(0, 1)

    plt.show()
