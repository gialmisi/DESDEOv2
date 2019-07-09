"""Define different solvers to solve MOO problems of various kind.

"""
import abc
import logging
import logging.config
from abc import abstractmethod
from os import path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution

from desdeo.problem.Problem import ProblemBase
from desdeo.solver.ASF import ASFBase

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class SolverError(Exception):
    """Raised when an error related to the solver classes in
    encountered.

    """

    pass


class SolverBase(abc.ABC):
    """A base class to define the interface for the various solver classes.

    """

    @abstractmethod
    def solve(
        self, args: Any
    ) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Solves the problem and returns relevant information.

        Args:
            *args(Any): Any kind of arguments that are relevant to the solver.

        Returns:
            Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]: A tuple
            containing the decision variables as the first element and the
            evaluation result of the underlaying porblem as the second element.

        """
        pass


class WeightingMethodSolver(SolverBase):
    """A class to represent a solver for solving a problems using the
    weighting method.

    Args:
        problem (ProblemBase): The underlaying problem obejest with the
        specifications of the problem to solve.
        weights (np.ndarray): The weights corresponsing to each objectie in the
        problem.

    Attributes:
        problem (ProblemBase): The underlaying problem obejest with the
        specifications of the problem to solve.
        weights (np.ndarray): The weights corresponsing to each objectie in the
        problem.

    """

    def __init__(self, problem: ProblemBase):
        self.__problem: ProblemBase = problem
        self.__weights: np.ndarray = None

    @property
    def problem(self) -> ProblemBase:
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

    def solve(
        self, weights: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Use differential evolution to solve the weighted sum problem.

        Args:
            weights (np.ndarray): Array of weights to weigh each objective in
            the sum.
        Returns:
            Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]: A tuple
            containing the decision variables as the first element and the
            evaluation result of the underlaying porblem as the second element.

        Note:
            This method might invoke runtime warnings, which are most likely
            related to the usage of infinity (np.inf).

        """
        self.__weights = weights

        func: Callable
        bounds: np.ndarray
        polish: bool
        results: OptimizeResult

        func = self._evaluator
        bounds = self.__problem.get_variable_bounds()
        polish = True

        results = differential_evolution(func, bounds, polish=polish)

        if results.success:
            decision_variables: np.ndarray = results.x

            return (
                decision_variables,
                self.__problem.evaluate(decision_variables),
            )
        else:
            logger.debug(results.message)
            raise SolverError(results.message)


class EpsilonConstraintSolver(object):
    """A class to represent a solver for solving porblems using the epsilon
    constraint method.

    Args:
        problem (ProblemBase): The underlaying problem obeject with the
        specifications of the problem to solve
        epsilons (np.ndarray): The epsilon values to set as the upper limit
        for each objective when treated as a constraint.

    Attributes:
        problem (ProblemBase): The underlaying problem obeject with the
        specifications of the problem to solve
        epsilons (np.ndarray): The epsilon values to set as the upper limit
        for each objective when treated as a constraint.

    Note:
        To be implemented.

    """

    def __init__(self, problem: ProblemBase, epsilons: np.ndarray):
        self.__problem: ProblemBase = problem
        self.__epsilons: np.ndarray = epsilons

    @property
    def problem(self) -> ProblemBase:
        return self.__problem

    @property
    def epsilons(self) -> np.ndarray:
        return self.__epsilons

    @epsilons.setter
    def epsilons(self, val: np.ndarray):
        self.__epsilons = val

    def _evaluator(self, decision_vectors: np.ndarray) -> np.ndarray:
        pass

    def solve(self, epsilons: np.ndarray):
        pass


class ASFSolver(SolverBase):
    """A class to represent a solver tha uses the achievement scalarizing
    method to solve a multiobjective optimization prblem.

    Args:
        problem (ProblemBase): The underlaying problem obejest with the
        specifications of the problem to solve.

    Attributes:
        problem (ProblemBase): The underlaying problem obejest with the
        specifications of the problem to solve.

    """

    def __init__(self, problem: ProblemBase):
        self.__problem: ProblemBase = problem

    def solve(
        self, asf: ASFBase
    ) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        pass
