import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D  # noqa

from desdeo.methods.InteractiveMethod import InteractiveMethodError
from desdeo.methods.Nautilus import ENautilus
from desdeo.problem.Problem import ScalarDataProblem


logging.getLogger("matplotlib").setLevel(logging.WARNING)


def test_initialization(sphere_pareto):
    method = ENautilus()
    xs, fs = sphere_pareto
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
    method = ENautilus()
    xs, fs = sphere_pareto
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
    method = ENautilus()
    xs, fs = sphere_pareto
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
    method = ENautilus()
    xs, fs = sphere_pareto
    total_iter = 10
    nadir, ideal = method.initialize(total_iter, 5, xs, fs)

    for i in range(total_iter - 1):
        # till the penultimate iteration
        zs, f_lows = method.iterate()
        r = np.random.randint(0, 5)
        method.interact(zs[r], f_lows[r])

    r = np.random.randint(0, 5)
    _, res = method.interact(zs[r], f_lows[r])

    assert np.any(np.all(np.isclose(res, fs), axis=1))


def test_not_enough_points(sphere_pareto):
    method = ENautilus()
    idxs = np.random.randint(0, len(sphere_pareto[0]), size=5)
    xs, fs = sphere_pareto[0][idxs], sphere_pareto[1][idxs]

    _, _ = method.initialize(10, 10, xs, fs)
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
    method = ENautilus()
    xs, fs = sphere_pareto[0][0:500], sphere_pareto[1][0:500]

    _, _ = method.initialize(10, 10, xs, fs)

    while method.ith > 1:
        zs, lows = method.iterate()
        method.interact(zs[0], lows[0])

    last_zs, last_lows = zs, lows

    x, f = method.interact(zs[0], lows[0])

    method.iterate()
    method.iterate()

    much_zs, much_lows = method.iterate()

    assert np.all(np.isclose(last_zs, much_zs))
    assert np.all(np.isclose(last_lows, much_lows))

    much_x, much_f = method.interact(last_zs[0], last_lows[0])

    assert np.all(np.isclose(x, much_x))
    assert np.all(np.isclose(f, much_f))


def test_dataproblem(sphere_pareto):
    xs, fs = sphere_pareto
    data_prob = ScalarDataProblem(xs, fs)
    method_data = ENautilus(data_prob)
    method_expl = ENautilus()

    method_data.initialize(10, 7)
    method_expl.initialize(10, 7, xs, fs)

    np.random.seed(1)
    zs_data, lows_data = method_data.iterate()
    np.random.seed(1)
    zs_expl, lows_expl = method_expl.iterate()

    assert np.all(np.isclose(zs_data, zs_expl))
    assert np.all(np.isclose(lows_data, lows_expl))

    for i in range(3):
        np.random.seed(1)
        method_data.interact(zs_data[i], lows_data[i])
        zs_data, lows_data = method_data.iterate()

        np.random.seed(1)
        method_expl.interact(zs_expl[i], lows_expl[i])
        zs_expl, lows_expl = method_expl.iterate()

    assert np.all(np.isclose(zs_data, zs_expl))
    assert np.all(np.isclose(lows_data, lows_expl))

    np.random.seed(None)

    with pytest.raises(InteractiveMethodError):
        method2 = ENautilus()
        method2.initialize(5, 10)
