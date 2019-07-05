"""Define different solvers to solve MOO problems of various kind.

"""

import abc
from abc import abstractmethod
from typing import Any

import numpy as np

from desdeo.problem.Problem import ScalarMOProblem


class SolverBase(abc.ABC):
    """A base class to define the interface for the various solver classes.

    """

    @abstractmethod
    def solve(self, args: Any) -> Any:
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

    def _evaluator(
        self, decision_vectors: np.ndarray, weight_vector: np.ndarray
    ) -> np.ndarray:
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
            each solution evaluated using the decision variable vectors
            present in decision_vectors.

        Note:
            If any of the constraints are broken for a particular solution, the
            corresponding weighted sum is set to infinity. It is therefore
            assumed that each objective is to be minimized.

        """
        objective_vectors: np.ndarray
        constraint_vectors: np.ndarray
        objective_vectors, constraint_vectors = self.__problem.evaluate(
            decision_vectors
        )

        weighted_sums = np.zeros(len(objective_vectors))
        for ind, elem in enumerate(objective_vectors):
            if np.any(constraint_vectors[ind] < 0):
                # If any of the constraints is broken, set the sum to
                # infinity
                weighted_sums[ind] = np.Inf
            else:
                # Compute the weigted sum for each objective vector using the
                # dot product
                weighted_sums[ind] = np.dot(
                    objective_vectors[ind], weight_vector
                )

        return weighted_sums

    def solve(self, decision_vector: np.ndarray) -> Any:
        pass
