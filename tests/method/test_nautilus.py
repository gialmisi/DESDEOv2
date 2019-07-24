import numpy as np
import pytest
from pytest import approx

from desdeo.methods.InteractiveMethod import InteractiveMethodError
from desdeo.methods.Nautilus import Nautilus


def test_calculate_iteration_point(NautilusInitializedRiver):
    method = NautilusInitializedRiver

    assert method.h == 0

    assert method.ith == 5
    ith = 5

    assert method.zs[method.h] is not None
    z_prev = np.array([2.5, -3.4, 5, 12])
    method.zs[method.h] = z_prev

    assert method.zs[method.h + 1] is None

    assert method.fs[method.h + 1] is None
    f_h = np.array([1.1, -5.2, 1, 8])
    method.fs[method.h + 1] = f_h

    expected = ((ith - 1) / (ith)) * z_prev + (1 / (ith)) * f_h

    actual = method._calculate_iteration_point()
    np.all(np.isclose(expected, actual))


def test_calculate_distance(NautilusInitializedRiver):
    method = NautilusInitializedRiver

    assert method.h == 0
    h = 0

    z_h = np.array([2.5, 3.5, 5.0, -4.2])
    method.zs[h + 1] = z_h

    f_h = np.array([1.5, 2.7, 4.5, -5.4])
    method.fs[h + 1] = f_h

    nadir = np.array([10, 20, 30, 15])
    method.problem.nadir = nadir

    expected = 100 * np.linalg.norm(z_h - nadir) / np.linalg.norm(f_h - nadir)
    actual = method._calculate_distance()

    assert np.all(np.isclose(expected, actual))


def test_preference_index_set(NautilusInitializedRiver):
    method = NautilusInitializedRiver
    indices_good = np.array([1, 1, 2, 2])
    indices_bad_length = np.array([1, 2, 3])
    indices_bad_value = np.array([1, 5, 2, 3])

    method.preference_index_set = indices_good
    assert np.all(np.isclose(method.preference_index_set, indices_good))

    with pytest.raises(InteractiveMethodError):
        method.preference_index_set = indices_bad_length

    with pytest.raises(InteractiveMethodError):
        method.preference_index_set = indices_bad_value


def test_itn(NautilusInitializedRiver):
    method = NautilusInitializedRiver

    method.itn = 13
    assert method.itn == 13

    with pytest.raises(InteractiveMethodError):
        method.itn = -1


def test_ith(NautilusInitializedRiver):
    method = NautilusInitializedRiver

    method.ith = 3
    assert method.ith == 3

    with pytest.raises(InteractiveMethodError):
        method.ith = -1

    with pytest.raises(InteractiveMethodError):
        method.ith = 20


def test_preference_percentages(NautilusInitializedRiver):
    method = NautilusInitializedRiver
    percentages_good = np.array([10, 20, 30, 40])
    percentages_bad_len = np.array([50, 50])
    percentages_bad_sum = np.array([10, 10, 10, 20])

    method.preference_percentages = percentages_good
    assert np.all(np.isclose(method.preference_percentages, percentages_good))

    with pytest.raises(InteractiveMethodError):
        method.preference_percentages = percentages_bad_len

    with pytest.raises(InteractiveMethodError):
        method.preference_percentages = percentages_bad_sum


def test_nautilus_initialization(RiverPollutionProblem):
    problem = RiverPollutionProblem
    method = Nautilus(problem)
    z, bounds, dist = method.initialize(10)

    assert method.itn == 10

    assert len(method.problem.ideal) == 4
    assert len(method.problem.nadir) == 4

    assert method.h == 0
    assert method.ith == method.itn

    assert len(method.zs) == method.itn + 1
    assert np.all(np.isclose(method.zs[0], method.problem.nadir))

    assert len(method.lower_bounds) == method.itn + 1
    assert np.all(np.isclose(method.lower_bounds[0], method.problem.ideal))

    assert len(method.upper_bounds) == method.itn + 1
    assert np.all(np.isclose(method.upper_bounds[0], method.problem.nadir))

    assert len(method.xs) == method.itn + 1
    assert len(method.fs) == method.itn + 1
    assert len(method.ds) == method.itn + 1

    assert np.all(np.isclose(method.asf.nadir_point, method.problem.nadir))
    assert np.all(np.isclose(method.asf.utopian_point, method.problem.ideal))

    assert np.all(np.isclose(z, method.problem.nadir))
    assert dist == approx(0)
    assert np.all(
        np.isclose(
            bounds,
            np.array(list(zip(method.problem.ideal, method.problem.nadir))),
        )
    )


def test_nautilus_interact_after_initialization(NautilusInitializedRiver):
    method = NautilusInitializedRiver
    percentages = np.array([10, 20, 30, 40])

    # current iteration and iterations left should not change on first interact
    res = method.interact(percentages=percentages)
    assert method.h == 0
    assert method.ith == 5
    assert res == 5

    # on other iterations, h should increase and ith should decrease by one
    method.ith = 4
    res = method.interact(percentages=percentages)
    assert method.h == 1
    assert method.ith == 3
    assert res == 3

    # on last iteration, return the decision variables and objectives
    # of the final solution
    method.ith = 1
    res = method.interact(percentages=percentages)
    assert res == (None, None)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nautilus_iterate(RiverPollutionProblem):
    """Iterat once and compare the results to the example in the original
    article

    """
    problem = RiverPollutionProblem
    problem.nadir = np.array([-4.07, -2.83, -0.32, 9.71])
    problem.ideal = np.array([-6.34, -3.44, -7.50, 0.00])
    method = Nautilus(problem)
    method.asf.roo = 0.000001
    z0, _, _ = method.initialize(3)

    # first interaction
    index_set = np.array([2, 2, 1, 1])
    remaining = method.interact(index_set=index_set)
    assert remaining == 3
    assert method.h == 0

    # first iteration
    z1, bounds, dist1 = method.iterate()
    f1 = method.fs[method.h + 1]
    z2_lo = np.array([low for (low, _) in bounds])

    # compared to the article's example
    z1_article = np.array([-4.81, -3.00, -1.32, 8.35])
    f1_article = np.array([-6.28, -3.34, -3.32, 5.65])
    z2_lo_article = np.array([-6.33, -3.42, -7.49, 0.65])

    assert np.all(np.isclose(z1, z1_article, atol=0.1))
    assert np.all(np.isclose(f1, f1_article, atol=0.1))
    assert np.all(np.isclose(z2_lo, z2_lo_article, atol=0.1))
    assert dist1 == approx(100 / 3)

    # second interaction
    remaining = method.interact(use_previous_preference=True)
    assert remaining == 2
    assert method.h == 1

    # second iteration
    z2, bounds, dist2 = method.iterate()
    f2 = method.fs[method.h + 1]
    z3_lo = np.array([low for (low, _) in bounds])

    z2_article = np.array([-5.54, -3.17, -2.32, 7.00])
    f2_article = f1_article
    z3_lo_article = np.array([-6.31, -3.39, -7.15, 2.28])

    assert np.all(np.isclose(z2, z2_article, atol=0.1))
    assert np.all(np.isclose(f2, f2_article, atol=0.1))
    assert np.all(np.isclose(z3_lo, z3_lo_article, atol=0.1))
    assert dist2 == approx(2 * 100 / 3)

    # third interaction
    index_set = np.array([2, 3, 1, 4])
    remaining = method.interact(index_set=index_set, step_back=True)

    assert remaining == 2
    assert method.h == 1

    # third iteration
    z3, bounds, dist3 = method.iterate()
    f3 = method.fs[method.h + 1]
    z4_lo = np.array([low for (low, _) in bounds])

    z3_article = np.array([-5.56, -3.12, -1.79, 5.82])
    f3_article = np.array([-6.31, -3.24, -2.26, 3.29])
    z4_lo_article = np.array([-6.32, -3.35, -7.14, 1.68])

    assert np.all(np.isclose(z3, z3_article, atol=0.1))
    assert np.all(np.isclose(f3, f3_article, atol=0.1))
    assert np.all(np.isclose(z4_lo, z4_lo_article, atol=0.1))
    assert dist3 == approx(62, abs=1)

    # fourth interaction
    index_set = np.array([1, 2, 1, 2])
    remaining = method.interact(index_set=index_set)

    assert remaining == 1
    assert method.h == 2

    # fourth iteration
    z4, bounds, dist4 = method.iterate()
    f4 = method.fs[method.h + 1]
    z5_lo = np.array([low for (low, _) in bounds])

    z4_article = np.array([-6.30, -3.26, -2.60, 3.63])
    f4_article = z4_article

    assert np.all(np.isclose(z4, z4_article, atol=0.1))
    assert np.all(np.isclose(f4, f4_article, atol=0.1))
    assert np.all(np.equal(z5_lo, None))
    assert dist4 == approx(100, abs=1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nautilus_short_step(NautilusInitializedRiver):
    method = NautilusInitializedRiver
    percentages = np.array([20, 20, 30, 30])

    method.interact(percentages=percentages)
    z1, bounds1, dist1 = method.iterate()

    method.interact(use_previous_preference=True)
    z2, bounds2, dist2 = method.iterate()
    low2 = np.array([low for (low, _) in bounds2])
    high2 = np.array([high for (_, high) in bounds2])

    method.interact(step_back=True, short_step=True)
    z3, bounds3, dist3 = method.iterate()
    low3 = np.array([low for (low, _) in bounds3])
    high3 = np.array([high for (_, high) in bounds3])

    assert np.all(np.isclose(0.5 * z1 + 0.5 * z2, z3))
    # The high bounds shouldn't change, it's always the nadir
    assert np.all(np.isclose(high2, high3))
    assert np.all(low2 > low3)
    assert dist2 > dist3
