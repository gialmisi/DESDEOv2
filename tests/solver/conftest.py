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
            lambda x: -(2 * np.pi * x[0] ** 2 + 2 * x[0] * x[1]),
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
