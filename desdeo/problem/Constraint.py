""" Here different kinds of constraint functions are defined.

"""

import logging
import logging.config
from os import path
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

log_conf_path = path.join(path.dirname(path.abspath(__file__)),
                          "../../logger.cfg")
logging.config.fileConfig(fname=log_conf_path,
                          disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class ConstraintError(Exception):
    """Raised when an error related to the Constraint class in encountered.

    """


class ConstraintBase(ABC):
    """Base class for constraints.

    """

    @abstractmethod
    def evaluate(self,
                 decision_vector: np.ndarray,
                 objective_vector: np.ndarray) -> float:
        """Evaluate the constraint functions and return a float
        indicating how severely the constraint has been broken.

        Args:
            decision_vector (np.ndarray): A vector containing the decision
            variable values.
            objective_vector (np.ndarray): A vector containing the objective
            function values.

        Returns:
            float: A float representing how and if the constraing has
            been violated. A positive value represents no violation and a
            negative value represents a violation. The absolute value of the
            returned float functions as an indicator of the severity of the
            violation (or how well the constraint holds, if the returned value
            of positive).

        """
        pass


class ScalarConstraint(ConstraintBase):
    """A simple scalar constraint that evaluates to a single scalar.

    Args:
        name (str): Name of the constraint.
        n_decision_vars (int): Number of decision variables present in the
        constraint.
        n_objective_funs (int): Number of objective functions present in
        the constraint.
        evaluator (Callable): A callable to evaluate the constraint.

    Attributes:
        name (str): Name of the constraint.
        n_decision_vars (int): Number of decision variables present in the
        constraint.
        n_objective_funs (int): Number of objective functions present in
        the constraint.
        evaluator (Callable): A callable to evaluate the constraint.

    """
    def __init__(self,
                 name: str,
                 n_decision_vars: int,
                 n_objective_funs: int,
                 evaluator: Callable) -> None:
        self.__name: str = name
        self.__n_decision_vars: int = n_decision_vars
        self.__n_objective_funs: int = n_objective_funs
        self.__evaluator: Callable = evaluator

    @property
    def name(self) -> str:
        return self.__name

    @property
    def n_decision_vars(self) -> int:
        return self.__n_decision_vars

    @property
    def n_objective_funs(self) -> int:
        return self.__n_objective_funs

    @property
    def evaluator(self) -> Callable:
        return self.__evaluator

    def evaluate(self,
                 decision_vector: np.ndarray,
                 objective_vector: np.ndarray) -> float:
        """Evaluate the constraint and return a float indicating how and if the
        constraint was violated. A negative value indicates a violation and
        a positive value indicates a non-violation.

        Args:
            decision_vector (np.ndarray): A vector containing the values of
            the decision variables.
            objective_vector (np.ndarray): A vector containing the values
            of the objective functions.

        Returns:
            float: A float indicating how the constraint holds.

        """
        if len(decision_vector) != self.__n_decision_vars:
            msg = ("Decision vector {} is of wrong lenght: "
                   "Should be {}, but is {}").format(
                       decision_vector,
                       self.__n_decision_vars,
                       len(decision_vector))
            logger.debug(msg)
            raise ConstraintError(msg)

        if len(objective_vector) != self.__n_objective_funs:
            msg = ("Objective vector {} is of wrong lenght:"
                   " Should be {}, but is {}").format(
                       objective_vector,
                       self.__n_objective_funs,
                       len(objective_vector))
            logger.debug(msg)
            raise ConstraintError(msg)

        try:
            result = self.__evaluator(decision_vector, objective_vector)
        except (TypeError, IndexError) as e:
            msg = ("Bad arguments {} and {} supllied to the evaluator:"
                   " {}").format(
                       str(decision_vector),
                       objective_vector,
                       str(e))
            raise ConstraintError(msg)

        return result
