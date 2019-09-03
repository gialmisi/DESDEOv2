"""Implementation of solvers to find certain set of points.

"""
import abc
import logging
import logging.config
from abc import abstractmethod
from os import path
from typing import Tuple

import numpy as np

from desdeov2.problem.Problem import ProblemBase
from desdeov2.solver.NumericalMethods import NumericalMethodBase

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class PointSolverError(Exception):
    """Raised when an error related to the PointSolvers is encountered.

    """

    pass


class PointSolverBase(abc.ABC):
    """A base class that solve for particular points (or group of points) that
    solve a MOO problem, or satisfy other conditions.

    """

    @abstractmethod
    def solve(self) -> np.ndarray:
        """Solves for the points and returns them in an array."""
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

    def __init__(self, problem: ProblemBase, method: NumericalMethodBase):
        self.__problem = problem
        self.__method = method

    @property
    def problem(self) -> ProblemBase:
        return self.__problem

    @problem.setter
    def problem(self, val: ProblemBase):
        self.__problem = val

    @property
    def method(self) -> NumericalMethodBase:
        return self.__method

    @method.setter
    def method(self, val: NumericalMethodBase):
        self.__method = val

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
        objective_vals, constraint_vals = self.problem.evaluate(
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
        pay_off_table = np.zeros(
            (self.__problem.n_of_objectives, self.problem.n_of_objectives)
        )
        func = self._evaluator
        bounds = self.problem.get_variable_bounds()

        for ind in range(self.problem.n_of_objectives):
            args = (ind,)
            x = self.method.run(func, bounds, args)

            pay_off_table[ind], _ = self.problem.evaluate(x)

        # The ideal point can be found on the diagonal of the PO-table,
        # and an estimate of the nadir by taking the maximun value of each
        # column.
        return np.array(
            [pay_off_table.diagonal(), np.amax(pay_off_table, axis=0)]
        )
