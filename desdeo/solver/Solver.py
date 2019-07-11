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
    """Raised when an error related to the solver classes is
    encountered.

    """

    pass


class SolverBase(abc.ABC):
    """A base class to define the interface for the various solver classes.

    Args:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    """

    def __init__(self, problem: ProblemBase):
        self.__problem = problem

    @property
    def problem(self) -> ProblemBase:
        return self.__problem

    @abstractmethod
    def solve(
        self, args: Any
    ) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Solves the problem and returns relevant information.

        Args:
            *args(Any): Any kind of arguments that are relevant to the solver.
            See the solvers for further details.

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
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.
        weights (np.ndarray): The weights corresponsing to each objectie in the
        problem.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.
        weights (np.ndarray): The weights corresponsing to each objectie in the
        problem.

    """

    def __init__(self, problem: ProblemBase):
        super().__init__(problem)
        self.__weights: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @weights.setter
    def weights(self, val: np.ndarray):
        self.__weights = val

    def _evaluator(self, decision_vectors: np.ndarray) -> np.ndarray:
        """A helper function to transform the output of the MOO problem
        to a weighted sum.

        Args:
            decision_vectors (np.ndarray): An array of arrays representing
            decision variable values to evaluate the MOO problem.

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
        objective_vectors, constraint_vectors = self.problem.evaluate(
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
            evaluation results of the underlaying porblem as the second
            element.

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
        bounds = self.problem.get_variable_bounds()
        polish = True

        results = differential_evolution(func, bounds, polish=polish)

        if results.success:
            decision_variables: np.ndarray = results.x

            return (
                decision_variables,
                self.problem.evaluate(decision_variables),
            )
        else:
            logger.debug(results.message)
            raise SolverError(results.message)


class EpsilonConstraintSolver(SolverBase):
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
        super().__init__(problem)
        self.__epsilons: np.ndarray = epsilons

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
    method to solve a multiobjective optimization problem.

    Args:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    """

    def __init__(self, problem: ProblemBase):
        super().__init__(problem)
        self.__asf: ASFBase
        self.__reference_point: np.ndarray

    @property
    def asf(self) -> ASFBase:
        return self.__asf

    @asf.setter
    def asf(self, val: ASFBase):
        self.__asf = val

    @property
    def reference_point(self) -> np.ndarray:
        return self.__reference_point

    @reference_point.setter
    def reference_point(self, val: np.ndarray):
        self.__reference_point = val

    def _evaluator(self, decision_vector: np.ndarray) -> np.ndarray:
        """A helper function to express the original problem as an ASF
           problem.

        Args:
            decision_vectors (np.ndarray): An array of arrays representing the
            decision variable values to evaluate the underlaying MOO problem.

        Returns:
            np.ndarray: An array of the ASF problem values corresponding to
            each decision variable vector.

        Note:
            Requires an achievement scalarizing function to be defined and set
            to the ASFSolver object. See ASF.py for available functions and
            further details.

        """

        objective_vectors: np.ndarray
        constraints: np.ndarray
        asf_values: np.ndarray

        objective_vectors, constraints = self.problem.evaluate(decision_vector)
        asf_values = np.zeros(len(objective_vectors))

        for ind, elem in enumerate(objective_vectors):
            if np.any(constraints[ind] < 0):
                # suicide method for broken constraints
                asf_values[ind] = np.inf
            else:
                asf_values[ind] = self.__asf(elem, self.__reference_point)

        return asf_values

    def solve(
        self, reference_point: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Use differential evolution to solve the ASF problem.

        Args:
            reference_point (np.ndarray): An array representing the reference
            point in the objective function space used the is ASF.
        Returns:
            Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]: A tuple
            containing the decision variables as the first element and the
            evaluation results of the underlaying porblem as the second
            element.

        """
        self.__reference_point = reference_point

        func: Callable
        bounds: np.ndarray
        polish: bool
        results: OptimizeResult

        func = self._evaluator
        bounds = self.problem.get_variable_bounds()
        polish = True

        results = differential_evolution(func, bounds, polish=polish)

        if results.success:
            decision_variables: np.ndarray = results.x

            return (
                decision_variables,
                self.problem.evaluate(decision_variables),
            )
        else:
            logger.debug(results.message)
            raise SolverError(results.message)
