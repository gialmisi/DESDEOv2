"""This is for purely testing.

"""

import abc
from abc import abstractmethod

from desdeo.problem.Problem import ScalarMOProblem
from desdeo.problem.Objective import ScalarObjective
from desdeo.problem.Constraint import (constraint_function_factory,
                                       ScalarConstraint)
from desdeo.problem.Variable import Variable

import numpy as np

from typing import Any, Callable


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

res = problem.evaluate(np.array([[5, 15], [5, 5], [15, 25]]))


class SolverBase(abc.ABC):
    """A base class to define the interface for the various solver classes.

    """
    @abstractmethod
    def solve(self, *args: Any) -> Any:
        """Solves the problem and returns relevant information.

        Args:
            *args(Any): Any kind of arguments that are relevant to the solver.

        Returns:
            Any: A solution of the solved problem and possibly auxillary data.
        """
        pass


class WeightingMethodSolver(SolverBase):
    """A class to represent a solver for solving a ScalarMOProblem using the
    weighting method.

    Args:
        problem (ScalarMOProblem): A ScalarMOProblem to be solved using the
        weighting method.

    """
    def __init__(self, problem: ScalarMOProblem):
        self.__problem: ScalarMOProblem = problem

    @property
    def problem(self):
        return self.__problem

    def _evaluator(self,
                   decision_vectors: np.ndarray,
                   weight_vector: np.ndarray) -> np.ndarray:
        """A helper function to transform the output of the ScalarMOProblem
        to weighted sum.

        Args:
            decision_vectors (np.ndarray): An array of arrays representing
            decision variable values to evaluate the ScalarMOProblem.
            weight_vector (np.ndarray): An array of weights to be used in the
            weighted sum returned by this function. It's length should match
            the number of objectives defined in ScalarMOProblem.

        Returns:
            np.ndarray: An array with the weighted sums corresponding to 
            each solution evaluated using the decision variable vectors present in
            decision_vectors.

        Note:
            If any of the constraints are broken for a particular solution, the 
            corresponding weighted sum is set to infinity. It is therefore assumed 
            that each objective is to be minimized.

        """
        objective_vectors: np.ndarray
        constraint_vectors: np.ndarray
        objective_vectors, constraint_vectors = \
            self.__problem.evaluate(decision_vectors)

        weighted_sums = np.zeros(len(objective_vectors))
        for ind, elem in enumerate(objective_vectors):
            if np.any(constraint_vectors[ind] < 0):
                # If any of the constraints is broken, set the sum to
                # infinity
                weighted_sums[ind] = np.Inf
            else:
                # Compute the weigted sum for each objective vector using the dot product
                weighted_sums[ind] = np.dot(
                    objective_vectors[ind],
                    weight_vector)

        return weighted_sums

    def solve(self, decision_vector: np.ndarray):
        return self._evaluator(decision_vector, np.array([1, 1, 1]))


solver = WeightingMethodSolver(problem)
print(problem.evaluate(np.array([[5, 15], [5, 5], [15, 25]])))
print(solver.solve(np.array([[5, 15], [5, 5], [15, 25]])))
