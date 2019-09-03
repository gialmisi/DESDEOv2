import os

import numpy as np
import pytest

from desdeov2.problem.Constraint import (
    ScalarConstraint,
    constraint_function_factory,
)
from desdeov2.problem.Objective import ScalarObjective
from desdeov2.problem.Problem import ScalarDataProblem, ScalarMOProblem
from desdeov2.problem.Variable import Variable


@pytest.fixture
def RiverPollutionProblem():
    """Return the river pollution problem as defined in `Miettinen 2010`_

    .. _Miettinen 2010:
        Miettinen, K.; Eskelinen, P.; Ruiz, F. & Luque, M.
        NAUTILUS method: An interactive technique in multiobjective
        optimization based on the nadir point
        Europen Joural of Operational Research, 2010, 206, 426-434

    """
    # Variables
    variables = []
    variables.append(Variable("x_1", 0.5, 0.3, 1.0))
    variables.append(Variable("x_2", 0.5, 0.3, 1.0))

    # Objectives
    objectives = []
    objectives.append(ScalarObjective("f_1", lambda x: -4.07 - 2.27 * x[0]))

    objectives.append(
        ScalarObjective(
            "f_2",
            lambda x: -2.60
            - 0.03 * x[0]
            - 0.02 * x[1]
            - 0.01 / (1.39 - x[0] ** 2)
            - 0.30 / (1.39 - x[1] ** 2),
        )
    )

    objectives.append(
        ScalarObjective("f_3", lambda x: -8.21 + 0.71 / (1.09 - x[0] ** 2))
    )

    objectives.append(
        ScalarObjective("f_4", lambda x: -0.96 + 0.96 / (1.09 - x[1] ** 2))
    )

    # problem
    problem = ScalarMOProblem(objectives, variables, [])

    return problem


@pytest.fixture
def CylinderProblem():
    """Return a simple cylinder ScalarMOProblem.

    Note:
        In this problem consider a cell shaped like a cylinder with a circular
        cross-section.

        The shape of the cell is here determined by two quantities, its radius
        `r` and its height `h`. We want to maximize the volume of the cylinder
        and minimize the surface area. In addition to this, cylinder's height
        should be close to 15 units, i.e. we minimize the absolute difference
        between the height and 15.

        Finally the cylinder's height must be greater or equal to its
        width. Thus there are 2 decision variables, 3 objectives and 1
        constraint in this problem.
    """
    # Variables r := x[0], h := x[1]
    variables = []
    variables.append(Variable("radius", 10, 5, 15))
    variables.append(Variable("height", 10, 5, 25))

    # Objectives
    objectives = []
    objectives.append(
        ScalarObjective("Volume", lambda x: np.pi * x[0] ** 2 * x[1])
    )
    objectives.append(
        ScalarObjective(
            "Surface area",
            lambda x: -(2 * np.pi * x[0] ** 2 + 2 * np.pi * x[0] * x[1]),
        )
    )
    objectives.append(
        ScalarObjective("Height Difference", lambda x: abs(x[1] - 15.0))
    )

    # Constraints
    constraints = []
    constraints.append(
        ScalarConstraint(
            "Height greater than width",
            len(variables),
            len(objectives),
            constraint_function_factory(
                lambda x, f: 2 * x[0] - x[1], 0.0, "<"
            ),
        )
    )

    # problem
    problem = ScalarMOProblem(objectives, variables, constraints)

    return problem


@pytest.fixture
def DTLZ1_3D():
    n = 3
    variables = []
    for i in range(n):
        variables.append(Variable("x{}".format(i), 0.5, 0, 1))

    def g(x):
        return 100 * (n - 2) + 100 * (
            np.sum((x[2:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[2:] - 0.5)))
        )

    objectives = []
    objectives.append(
        ScalarObjective("f1", lambda x: (1 + g(x)) * x[0] * x[1])
    )
    objectives.append(
        ScalarObjective("f2", lambda x: (1 + g(x)) * x[0] * (1 - x[1]))
    )
    objectives.append(ScalarObjective("f3", lambda x: (1 + g(x)) * (1 - x[0])))

    constraints = []
    for i in range(n):
        constraints.append(
            ScalarConstraint(
                "",
                len(variables),
                len(objectives),
                constraint_function_factory(lambda x, f: x[i], 0.0, ">"),
            )
        )
        constraints.append(
            ScalarConstraint(
                "",
                len(variables),
                len(objectives),
                constraint_function_factory(lambda x, f: x[i], 1.0, "<"),
            )
        )

    problem = ScalarMOProblem(objectives, variables, constraints)
    return problem


@pytest.fixture
def sphere_pareto():
    """Return a tuple of points representing the angle parametrized surface of
    a sphere's positive octant.  The first element represents the theta and phi
    angle values and the second the corresponding cartesian (x,y,z)
    coordinates

    """
    dirname = os.path.dirname(__file__)
    relative = "../data/pareto_front_3d_sphere_1st_octant_surface.dat"
    filename = os.path.join(dirname, relative)
    p = np.loadtxt(filename)
    return (p[:, :2], p[:, 2:])


@pytest.fixture
def simple_data():
    xs = np.array([[1, 2, 3], [2, 2, 2], [2.2, 3.3, 6.6], [-1.05, -2.05, 3.1]])
    fs = np.array([[np.sum(x), np.linalg.norm(x)] for x in xs])
    return xs, fs


@pytest.fixture
def simple_data_problem(simple_data):
    xs, fs = simple_data
    return ScalarDataProblem(xs, fs)


@pytest.fixture
def four_dimenional_data_with_extremas():
    """Four dimensional data, both in the objective and decision space. Also
    returns nadir and ideal points.

    """
    xs = np.array(
        [
            [5.2, 3.3, 9.2, -3.1],
            [-9.1, 6.5, -3.3, -1.1],
            [-1.2, -2.1, 1.2, 2.1],
            [9.9, -9.9, 0.4, 0.1],
        ]
    )

    fs = np.array(
        [
            [-1.5, -8.8, 8.5, 1.2],
            [0.2, -9.9, 2.1, 3.4],
            [1.5, 1.2, 1.2, 9.4],
            [6.4, 8.5, 7.4, 0.2],
        ]
    )

    nadir = np.array([7, 10, 8, 10])
    ideal = np.array([-1.5, -9.9, 1.2, 0.2])

    return xs, fs, nadir, ideal
