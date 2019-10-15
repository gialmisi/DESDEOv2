import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D  # noqa

from desdeov2.methods.InteractiveMethod import InteractiveMethodError
from desdeov2.methods.Nautilus import ENautilus
from desdeov2.problem.Problem import ScalarDataProblem

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def test_initialization(sphere_pareto):
    xs, fs = sphere_pareto
    data_prob = ScalarDataProblem(xs, fs)
    method = ENautilus(data_prob)
    nadir, ideal = method.initialize(10, 5, xs, fs)

    assert np.all(np.isclose(method.pareto_front, xs))
    assert np.all(np.isclose(method.objective_vectors, fs))

    assert np.all(nadir >= method.objective_vectors)
    assert np.all(ideal <= method.objective_vectors)

    assert method.n_iters == 10
    assert method.n_points == 5

    assert np.all(np.isnan(method.zshi))

    assert method.zshi.shape == (10, 5, 3)
    assert method.fhilo.shape == (10, 5, 3)
    assert method.d.shape == (10, 5)

    assert method.h == 0
    assert method.ith == 10

    assert np.all(np.isclose(method.obj_sub[0], method.objective_vectors))
    assert np.all(np.isclose(method.par_sub[0], method.pareto_front))

    assert np.all(np.isclose(method.zpref, method.nadir))

    # bad nadir
    with pytest.raises(InteractiveMethodError):
        method.nadir = np.array([1, 1])
    with pytest.raises(InteractiveMethodError):
        method.nadir = np.array([1, 1, 1, 1])

    # bad ideal
    with pytest.raises(InteractiveMethodError):
        method.ideal = np.array([1, 1])
    with pytest.raises(InteractiveMethodError):
        method.ideal = np.array([1, 1, 1, 1])

    # bad iters
    with pytest.raises(InteractiveMethodError):
        method.n_iters = -1

    # bad n points
    with pytest.raises(InteractiveMethodError):
        method.n_points = -1


def test_iterate(sphere_pareto):
    xs, fs = sphere_pareto
    data_prob = ScalarDataProblem(xs, fs)
    method = ENautilus(data_prob)
    nadir, ideal = method.initialize(10, 8, xs, fs)
    zs, fs = method.iterate()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.title(
        "test_iterate in test_enautilus.py:\n "
        "Green dots should be dominating the red dot.\n Green "
        "dots should be spread evenly and lay between the pareto\n "
        "front (blue dots) and the nadir (red dot).\n"
        "Close this to continue."
    )
    ax.scatter(
        method.obj_sub[0][:, 0],
        method.obj_sub[0][:, 1],
        method.obj_sub[0][:, 2],
        s=0.1,
        c="blue",
    )
    ax.scatter(
        method.zshi[0, :, 0],
        method.zshi[0, :, 1],
        method.zshi[0, :, 2],
        c="green",
    )
    ax.scatter(method.nadir[0], method.nadir[1], method.nadir[2], c="red")
    plt.show()


def test_interact_once(sphere_pareto):
    xs, fs = sphere_pareto
    data_prob = ScalarDataProblem(xs, fs)
    method = ENautilus(data_prob)
    nadir, ideal = method.initialize(10, 5, xs, fs)
    zs, fslow = method.iterate()

    method.interact(zs[0], fslow[0])

    assert method.h == 1
    assert method.ith == 9

    assert len(method.obj_sub[method.h]) <= len(method.obj_sub[method.h - 1])
    assert len(method.par_sub[method.h]) <= len(method.par_sub[method.h - 1])

    assert len(method.obj_sub[method.h]) == len(method.par_sub[method.h])

    with pytest.raises(InteractiveMethodError):
        # bad pref
        method.interact(np.array([1, 1]), np.array([1, 1, 1]))

    with pytest.raises(InteractiveMethodError):
        # bad pref
        method.interact(np.array([1, 1, 1, 1]), np.array([1, 1, 1]))

    with pytest.raises(InteractiveMethodError):
        # bad f_low
        method.interact(np.array([1, 1, 1]), np.array([1, 1]))

    with pytest.raises(InteractiveMethodError):
        # bad f_low
        method.interact(np.array([1, 1, 1]), np.array([1, 1, 1, 1]))


def test_interact_end(sphere_pareto):
    xs, fs = sphere_pareto
    data_prob = ScalarDataProblem(xs, fs)
    method = ENautilus(data_prob)
    xs, fs = sphere_pareto
    total_iter = 10
    nadir, ideal = method.initialize(total_iter, 5)

    for i in range(total_iter - 1):
        # till the penultimate iteration
        zs, f_lows = method.iterate()
        r = np.random.randint(0, 5)
        method.interact(zs[r], f_lows[r])

    r = np.random.randint(0, 5)
    _, res = method.interact(zs[r], f_lows[r])

    assert np.any(np.all(np.isclose(res, fs), axis=1))


def test_not_enough_points(sphere_pareto):
    idxs = np.random.randint(0, len(sphere_pareto[0]), size=5)
    xs, fs = sphere_pareto[0][idxs], sphere_pareto[1][idxs]
    data_prob = ScalarDataProblem(xs, fs)
    method = ENautilus(data_prob)

    _, _ = method.initialize(10, 10)
    zs, lows = method.iterate()

    zs_is_nans = np.isnan(zs)
    lows_is_nans = np.isnan(lows)

    assert np.all(np.equal(zs_is_nans, lows_is_nans))

    zs_points = 0
    for row in ~zs_is_nans:
        if np.all(row):
            zs_points += 1

    lows_points = 0
    for row in ~lows_is_nans:
        if np.all(row):
            lows_points += 1

    assert zs_points == lows_points


def test_iterate_too_much(sphere_pareto):
    xs, fs = sphere_pareto
    xs = xs[:500]
    fs = fs[:500]
    data_prob = ScalarDataProblem(xs, fs)
    method = ENautilus(data_prob)
    xs, fs = sphere_pareto
    xs = xs[:500]
    fs = fs[:500]

    _, _ = method.initialize(10, 10)

    while method.ith > 1:
        zs, lows = method.iterate()
        method.interact(zs[0], lows[0])

    last_zs, last_lows = method.iterate()
    last_x, last_f = method.interact(last_zs[0], last_lows[0])

    np.random.seed(1)
    method.iterate()
    np.random.seed(1)
    method.iterate()

    np.random.seed(1)
    much_zs, much_lows = method.iterate()

    # Compare NaN as equals since they just represent missing points.
    assert np.all(np.isclose(last_zs, much_zs, equal_nan=True))
    assert np.all(np.isclose(last_lows, much_lows, equal_nan=True))

    much_x, much_f = method.interact(much_zs[0], much_lows[0])
    print(much_x)
    print(last_x)
    assert np.all(np.isclose(last_x, much_x))
    assert np.all(np.isclose(last_f, much_f))


@pytest.mark.snipe
def test_article_example():
    """Tests the numerical example presented in the original article

    """
    data = np.loadtxt("./data/article_enautilus.dat")
    # convert the 1st and 3rd objectives to minimization objectives
    data = data * np.array([-1, 1, -1])

    # form the problem
    problem = ScalarDataProblem(data, data)
    method = ENautilus(problem)
    method.initialize(5, 6)

    # check the ideal and nadir, up to 2 decimals
    nadir_article = np.array([-408.49, 9.28, -22.13])
    ideal_article = np.array([-47526.37, 0.05, -100.00])

    nadir_rounded = np.around(method.nadir, 2)
    ideal_rounded = np.around(method.ideal, 2)

    assert np.allclose(nadir_article, nadir_rounded)
    assert np.allclose(ideal_article, ideal_rounded)

    # first iteration
    zs, lows = method.iterate()
    print("zs:", np.around(zs, 2))
