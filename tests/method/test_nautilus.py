import numpy as np

from desdeo.methods.Nautilus import Nautilus


def test_nautilus_initialization(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = Nautilus(problem)
    res = method.initialize(10)

    assert method.itn == 10

    assert len(method.problem.ideal) == 4
    assert len(method.problem.nadir) == 4

    assert method.h == 1
    assert method.ith == method.itn

    assert len(method.zs) == method.itn
    assert np.all(np.isclose(method.zs[0], method.problem.nadir))

    assert len(method.lower_bounds) == method.itn
    assert np.all(np.isclose(method.lower_bounds[1], method.problem.ideal))

    assert len(method.upper_bounds) == method.itn
    assert np.all(np.isclose(method.upper_bounds[1], method.problem.nadir))

    assert len(method.xs) == method.itn
    assert len(method.fs) == method.itn
    assert len(method.ds) == method.itn

    assert np.all(np.isclose(method.asf.nadir_point, method.problem.nadir))
    assert np.all(np.isclose(method.asf.utopian_point, method.problem.ideal))

    assert np.all(np.isclose(res, method.problem.nadir))


def test_calculate_iteration_point(NautilusInitializedRiver):
    method = NautilusInitializedRiver

    assert method.h == 1

    assert method.ith == 5
    ith = 5

    assert method.zs[method.h - 1] is not None
    z_prev = np.array([2.5, -3.4, 5, 12])
    method.zs[method.h - 1] = z_prev

    assert method.zs[method.h] is None

    assert method.fs[method.h] is None
    f_h = np.array([1.1, -5.2, 1, 8])
    method.fs[method.h] = f_h

    expected = ((ith - 1) / (ith)) * z_prev + (1 / (ith)) * f_h

    actual = method._calculate_iteration_point()
    np.all(np.isclose(expected, actual))


def test_calculate_distance(NautilusInitializedRiver):
    method = NautilusInitializedRiver

    assert method.h == 1
    h = 1

    assert method.zs[h] is None
    assert method.fs[h] is None
    assert method.problem.nadir is not None

    z_h = np.array([2.5, 3.5, 5.0, -4.2])
    method.zs[h] = z_h

    f_h = np.array([1.5, 2.7, 4.5, -5.4])
    method.fs[h] = f_h

    nadir = np.array([10, 20, 30, 15])
    method.problem.nadir = nadir

    expected = 100 * np.linalg.norm(z_h - nadir) / np.linalg.norm(f_h - nadir)
    actual = method._calculate_distance()

    assert np.all(np.isclose(expected, actual))
