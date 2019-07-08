"""Define different solvers to solve MOO problems of various kind.

"""

import abc
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution

from desdeo.problem.Problem import ScalarMOProblem


class SolverError(Exception):
    """Raised when an error related to the solver classes in
    encountered.

    """

    pass


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
        self.__weights: np.ndarray = None

    @property
    def problem(self) -> ScalarMOProblem:
        return self.__problem

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @weights.setter
    def weights(self, val: np.ndarray):
        self.__weights = val

    def _evaluator(self, decision_vectors: np.ndarray) -> np.ndarray:
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
            corresponding weighted sum is set to infinity, regardless of the
            weight value. It is therefore assumed that each objective is to be
            minimized.

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
                weighted_sums[ind] = np.inf
            else:
                # Compute the weigted sum for each objective vector using the
                # dot product
                weighted_sums[ind] = np.dot(
                    objective_vectors[ind], self.__weights
                )

        return weighted_sums

    def solve(self, weights: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Use differential evolution to solve the weighted sum problem.

        Args:
            weights (np.ndarray): Array of weights to weigh each objective in
            the sum.
        Returns:
            OptimizeResult: The results of the optimization. None, if the
            optimization if not successful.

        """
        self.__weights = weights

        func: Callable
        bounds: List[Tuple[float, float]]
        polish: bool
        results: OptimizeResult

        func = self._evaluator
        bounds = self.__problem.get_variable_bounds()
        polish = True

        results = differential_evolution(func, bounds, polish=polish)

        if results.success:
            decision_variables: np.ndarray = results.x
            function_value: float = results.fun

            return (decision_variables, function_value)
        else:
            # TODO add logger
            raise SolverError(results.message)
