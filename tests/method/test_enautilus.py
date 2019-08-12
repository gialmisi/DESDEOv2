import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D  # noqa

from desdeo.methods.InteractiveMethod import InteractiveMethodError
from desdeo.methods.Nautilus import ENautilus

logging.getLogger("matplotlib").setLevel(logging.WARNING)


@pytest.fixture
def sphere_pareto():
    """Return a tuple of points representing the angle parametrized surface of
    a sphere's positive octant.  The first element represents the theta and phi
    angle values and the second the corresponding cartesian (x,y,z)
    coordinates

    """
    dirname = os.path.dirname(__file__)
    relative = "../../data/pareto_front_3d_sphere_1st_octant_surface.dat"
    filename = os.path.join(dirname, relative)
    p = np.loadtxt(filename)
    return (p[:, :2], p[:, 2:])


def test_initialization(sphere_pareto):
    method = ENautilus()
    xs, fs = sphere_pareto
    nadir, ideal = method.initialize(xs, fs, 10, 5)

    assert np.all(np.isclose(method.pareto_front, xs))
    assert np.all(np.isclose(method.objective_vectors, fs))

    assert np.all(nadir >= method.objective_vectors)
    assert np.all(ideal <= method.objective_vectors)

    assert method.n_iters == 10
    assert method.n_points == 5

    assert np.all(np.isclose(method.zshi[0, :], method.nadir))

    assert method.zshi.shape == (10, 5, 3)
    assert method.fhilo.shape == (10, 5, 3)
    assert method.d.shape == (10, 5)

    assert method.h == 1
    assert method.ith == 10

    assert np.all(np.isclose(method.obj_sub[1], method.objective_vectors))
    assert np.all(np.isclose(method.par_sub[1], method.pareto_front))

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
    nadir, ideal = method.initialize(xs, fs, 10, 8)
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
        method.obj_sub[1][:, 0],
        method.obj_sub[1][:, 1],
        method.obj_sub[1][:, 2],
        s=0.1,
        c="blue",
    )
    ax.scatter(
        method.zshi[1, :, 0],
        method.zshi[1, :, 1],
        method.zshi[1, :, 2],
        c="green",
    )
    ax.scatter(method.nadir[0], method.nadir[1], method.nadir[2], c="red")
    plt.show()


def test_interact_once(sphere_pareto):
    method = ENautilus()
    xs, fs = sphere_pareto
    nadir, ideal = method.initialize(xs, fs, 10, 5)
    zs, fslow = method.iterate()

    method.interact(zs[0], fslow[0])

    assert method.h == 2
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
    nadir, ideal = method.initialize(xs, fs, total_iter, 5)

    for i in range(total_iter - 1):
        # till the penultimate iteration
        zs, f_lows = method.iterate()
        r = np.random.randint(0, 5)
        method.interact(zs[r], f_lows[r])

    r = np.random.randint(0, 5)
    _, res = method.interact(zs[r], f_lows[r])

    assert np.any(np.all(np.isclose(res, fs), axis=1))
