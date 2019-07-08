"""This is for purely testing.

"""

import abc
from abc import abstractmethod

from desdeo.problem.Problem import ScalarMOProblem
from desdeo.problem.Objective import ScalarObjective
from desdeo.problem.Constraint import (constraint_function_factory,
                                       ScalarConstraint)
from desdeo.problem.Variable import Variable
from desdeo.solver.Solver import WeightingMethodSolver


import numpy as np


# Variables r := x[0], h := x[1]
variables = []
variables.append(Variable("radius", 10, 5, 15))
variables.append(Variable("height", 10, 5, 25))

# Objectives
objectives = []
objectives.append(ScalarObjective("Volume", lambda x: np.pi * x[0]**2 * x[1]))
objectives.append(
    ScalarObjective(
        "Surface area", lambda x: -(2 * np.pi * x[0]**2 + 2 * x[0] * x[1])))
objectives.append(
    ScalarObjective("Height Difference", lambda x: abs(x[1] - 15.0)))

# Constraints
constraints = []
constraints.append(
    ScalarConstraint(
        "Height greater than width", len(variables), len(objectives),
        constraint_function_factory(lambda x, f: 2 * x[0] - x[1], 0.0, '<')))

# problem
problem = ScalarMOProblem(objectives, variables, constraints)

# solver
solver = WeightingMethodSolver(problem)
weights = np.array([1., 1., 1.0])
print(problem.get_variable_bounds())
print(solver.solve(weights))
