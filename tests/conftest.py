import numpy as np
import pytest

from desdeo.problem.Constraint import (
    ScalarConstraint,
    constraint_function_factory,
)
from desdeo.problem.Objective import ScalarObjective
from desdeo.problem.Problem import ScalarMOProblem
from desdeo.problem.Variable import Variable


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
