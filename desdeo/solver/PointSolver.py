import abc
import logging
import logging.config
from abc import abstractmethod
from os import path
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution

from desdeo.problem.Problem import ProblemBase

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class PointSolverBase(abc.ABC):
    """A base class that solve for particular points (or group of points) that
    solve a MOO problem, or satisfy other conditions.

    """

    @abstractmethod
    def solve(self) -> np.ndarray:
        """Solves for the points and returns them in an array."""
        pass


class PointSolverError(Exception):
    """Raised when an error related to the PointSolvers is encountered.

    """

    pass


class IdealAndNadirPointSolver(PointSolverBase):
    """Calculates the ideal and utopian points for a given problem. The ideal
    point is calculated by minimizing each function separately.

    Args:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    """

    def __init__(self, problem: ProblemBase):
        self.__problem = problem

    def _evaluator(
        self, decision_vector: np.ndarray, objective_index: int
    ) -> float:
        """A helper function to evaluate the problem and return the value of
        the specified objective.

        Arguments:
            decision_vector (np.ndarray): The decision variables to evaluate
            the problem with.
            objective_index (int): The index of the objective whose value is to
            returned.

        Returns:
            float: The value of the specified objective function.

        """
        objective_vals, constraint_vals = self.__problem.evaluate(
            decision_vector
        )
        if constraint_vals is not None and np.any(constraint_vals < 0):
            return np.inf
        else:
            return objective_vals[:, objective_index]

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the ideal and estimate of the nadir point for the MOO
        problem using a pay-off table.

        Returns:
            np.ndarray: An array with two elements. The first element is the
            ideal point specified by the diagonal of the generated pay-off
            table. The second element is an estimate of the nadir specified by
            taking the maximum value of each of the columns in the pay-off
            table.

        Note:
            The nadir point if an estimate, and therefore might be a bad
            representation of the real nadir point.
        """
        pay_off_table: np.ndarray
        func: Callable
        args: Tuple[int]
        tol: float
        bounds: np.ndarray
        polish: bool
        results: OptimizeResult

        pay_off_table = np.zeros(
            (self.__problem.n_of_objectives, self.__problem.n_of_objectives)
        )
        func = self._evaluator
        tol = 0.0001  # We want an accurate ideal point
        bounds = self.__problem.get_variable_bounds()
        polish = True

        for ind in range(self.__problem.n_of_objectives):
            args = (ind,)
            results = differential_evolution(
                func, bounds, args=args, tol=tol, polish=polish
            )

            if results.success:
                pay_off_table[ind], _ = self.__problem.evaluate(results.x)
            else:
                logger.debug(results.message)
                raise PointSolverError(results.message)

        # The ideal point can be found on the diagonal of the PO-table,
        # and an estimate of the nadir by taking the maximun value of each
        # column.
        return np.array(
            [pay_off_table.diagonal(), np.amax(pay_off_table, axis=0)]
        )
