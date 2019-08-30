import numpy as np
import pytest

from desdeo.methods.InteractiveMethod import InteractiveMethodError
from desdeo.methods.Nimbus import SNimbus
from desdeo.problem.Problem import ScalarDataProblem, ScalarMOProblem


@pytest.fixture
def sphere_nimbus(sphere_pareto):
    problem = ScalarDataProblem(*sphere_pareto)
    method = SNimbus(problem)
    return method


@pytest.fixture
def simple_nimbus(four_dimenional_data_with_extremas):
    xs, fs, nadir, ideal = four_dimenional_data_with_extremas
    problem = ScalarDataProblem(xs, fs)
    method = SNimbus(problem)
    return method


def test_no_pareto_given(sphere_nimbus):
    with pytest.raises(NotImplementedError):
        method_bad = SNimbus(ScalarMOProblem([], [], []))
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


def test_starting_point(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(3, starting_point=np.array([1.2, 3.2, 4.4]))

    assert np.all(np.isclose(method.current_point, np.array([1.2, 3.2, 4.4])))

    method.initialize(3)

    assert not np.all(
        np.isclose(method.current_point, np.array([1.2, 3.2, 4.4]))
    )


def test_nadir_and_ideal(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(3)

    assert np.all(np.isclose(method.nadir, [1, 1, 1]))
    assert np.all(np.isclose(method.ideal, [0, 0, 0]))


def test_first_iteration(sphere_nimbus):
    method = sphere_nimbus
    res_init = method.initialize(3)

    assert method.first_iteration
    res_iter = method.iterate()

    assert np.all(np.isclose(res_init, res_iter))
    assert not method.first_iteration


def test_bad_classificaitons(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(3, starting_point=np.array([2, 4, 3]))
    method.iterate()

    # wrong number of classificaiton
    with pytest.raises(InteractiveMethodError):
        method.interact(classifications=[("<", 0), ("=", 0)])

    # wrong type of classification
    with pytest.raises(InteractiveMethodError):
        method.interact(classifications=[("<", 0), ("=", 0), ("==", 0)])

    # wrong type of paramter for <=
    with pytest.raises(InteractiveMethodError):
        method.interact(classifications=[("<=", 3), ("=", 0), ("=", 0)])

    # wrong type of paramter for >=
    with pytest.raises(InteractiveMethodError):
        method.interact(classifications=[("<=", 1), ("=", 0), (">=", 2)])

    # bad classificaton
    with pytest.raises(InteractiveMethodError):
        method.interact(classifications=[("<=", 1), ("<", 0), ("<=", 2)])


def test_sort_classificaitons(sphere_nimbus):
    method = sphere_nimbus
    method.initialize(4, starting_point=np.array([3.2, 4.1, 5.5]))
    method.iterate()

    method.classifications = [("<", 0), ("<=", 3.5), ("0", 0)]
    method._sort_classsifications()

    assert method._SNimbus__ind_set_lt[0] == 0
    assert method._SNimbus__ind_set_lte[0] == 1
    assert method._SNimbus__ind_set_free[0] == 2

    assert method._SNimbus__aspiration_levels[0] == 3.5

    method.classifications = [("<", 0), (">=", 5.0), ("=", 0)]
    method._sort_classsifications()

    assert method._SNimbus__ind_set_lt[0] == 0
    assert method._SNimbus__ind_set_gte[0] == 1
    assert method._SNimbus__ind_set_eq[0] == 2

    assert method._SNimbus__upper_bounds[0] == 5.0


def test_create_reference_point(simple_nimbus):
    method = simple_nimbus
    method.initialize(4, starting_point=np.array([-1.5, -8.8, 8.5, 1.2]))

    method.iterate()

    method.interact(
        classifications=[(">=", 2.0), ("0", 0), ("<=", 5), ("0", 0)]
    )
    method._sort_classsifications()
    res1 = method._create_reference_point()
    assert res1[0] == 2.0
    assert res1[1] == method.nadir[1]
    assert res1[2] == 5.0
    assert res1[3] == method.nadir[3]

    method.interact(
        classifications=[("<", 0), ("=", 0), (">=", 14), (">=", 15)]
    )
    method._sort_classsifications()
    res2 = method._create_reference_point()
    assert res2[0] == method.ideal[0]
    assert res2[1] == method.current_point[1]
    assert res2[2] == 14.0
    assert res2[3] == 15.0


def test_archive_points(simple_nimbus):
    method = simple_nimbus
    method.initialize(4, starting_point=np.array([-1.5, -8.8, 8.5, 1.2]))

    assert method.archive == []
    ps1 = [(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))]

    method.interact(save_points=ps1)
    for (a, b) in zip(method.archive, ps1):
        assert np.all(np.isclose(a, b))

    ps2 = [
        (np.array([2, 1, 3]), np.array([0.2, 0.1, 0.3])),
        (np.array([-1, -2, -3]), np.array([-0.1, -0.2, -0.3])),
    ]

    method.interact(save_points=ps2)
    for (a, b) in zip(method.archive, ps1 + ps2):
        assert np.all(np.isclose(a, b))

    print(method.archive)


def test_interact_intermediate(simple_nimbus):
    method = simple_nimbus
    method.initialize(4, starting_point=np.array([-1.5, -8.8, 8.5, 1.2]))
    method.iterate()

    assert not method.generate_intermediate

    p1 = np.array([0.1, 0.2, 0.3, 0.4])
    p2 = np.array([11, 22, 33, 44])
    method.interact(search_between_points=(p1, p2))

    assert method.generate_intermediate

    with pytest.raises(InteractiveMethodError):
        method.interact(search_between_points=(p1))

    with pytest.raises(InteractiveMethodError):
        method.interact(search_between_points=(p1, p1, p2))


def test_create_intermediate_reference_points(simple_nimbus):
    method = simple_nimbus
    method.initialize(4, starting_point=np.array([-1.5, -8.8, 8.5, 1.2]))

    f1 = np.array([2, 5])
    f2 = np.array([5, 2])

    method.search_between_points = (f1, f2)
    method.n_intermediate_solutions = 3

    res = method._create_intermediate_reference_points()
    assert len(res) == method.n_intermediate_solutions

    assert np.all(np.isclose(res[0], f1 + [3 / 4, -3 / 4]))
    assert np.all(np.isclose(res[1], f1 + [3 / 2, -3 / 2]))
    assert np.all(np.isclose(res[2], f1 + [9 / 4, -9 / 4]))


def test_calculate_intermediate(simple_nimbus):
    method = simple_nimbus
    method.initialize(4, starting_point=np.array([-1.5, -8.8, 8.5, 1.2]))
    method.iterate()
    method.interact(
        classifications=[(">=", 2.0), ("0", 0), ("<=", 5), ("0", 0)]
    )
    _, fs, _ = method.iterate()
    method.interact(
        search_between_points=(fs[0], fs[1]), n_intermediate_solutions=5
    )
    xs, fs, arch = method.iterate()

    assert len(fs) == len(xs)
    assert len(fs) == 5


@pytest.mark.snipe
def test_iterate(simple_nimbus):
    method = simple_nimbus
    method.initialize(4, starting_point=np.array([-1.5, -8.8, 8.5, 1.2]))

    res_fst = method.iterate()
    assert np.all(np.isclose(res_fst, [-1.5, -8.8, 8.5, 1.2]))

    method.interact(
        classifications=[("<", 0), ("=", 0), (">=", 14), (">=", 15)]
    )

    res1_x, res1_f, archive1 = method.iterate()
    assert len(res1_x) == len(res1_f)
    assert len(res1_x) == 4
    assert len(archive1) == 0

    method.interact(
        most_preferred_point=res1_f[2],
        classifications=[("=", 0), ("<", 0), (">=", 14), ("0", 0)],
        n_generated_solutions=2,
        save_points=[(res1_x[1], res1_f[1])],
    )

    res2_x, res2_f, archive2 = method.iterate()
    assert len(res2_x) == 2
    assert len(res2_f) == 2
    assert len(archive2) == 1
    for (a, b) in zip(archive2, [(res1_x[1], res1_f[1])]):
        assert np.all(np.isclose(a, b))

    method.interact(
        search_between_points=(res2_f[0], archive2[0][1]),
        n_intermediate_solutions=6,
    )

    res3_x, res3_f, archive3 = method.iterate()

    assert len(res3_x) == len(res3_f)
    assert len(res3_x) == 6
    assert len(archive3) == 1
    for (a, b) in zip(archive2, [(res1_x[1], res1_f[1])]):
        assert np.all(np.isclose(a, b))

    method.interact(
        classifications=[("=", 0), ("0", 0), ("<=", 0.1), ("<", 0)],
        save_points=list(zip(res3_x, res3_f)),
        most_preferred_point=res3_f[3],
        n_generated_solutions=2,
    )

    res4_x, res4_f, archive4 = method.iterate()

    assert len(res4_x) == len(res4_f)
    assert len(res4_x) == 2
    assert len(archive4) == len(archive3) + len(list(zip(res3_x, res3_f)))
    for (a, b) in zip(archive4, archive3 + list(zip(res3_x, res3_f))):
        assert np.all(np.isclose(a, b))
