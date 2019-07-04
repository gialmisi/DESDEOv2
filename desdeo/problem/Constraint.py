""" Here different kinds of constraint functions are defined.

"""

import logging
import logging.config
from abc import ABC, abstractmethod
from os import path
from typing import Callable, List

import numpy as np

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class ConstraintError(Exception):
    """Raised when an error related to the Constraint class in encountered.

    """


class ConstraintBase(ABC):
    """Base class for constraints.

    """

    @abstractmethod
    def evaluate(
        self, decision_vector: np.ndarray, objective_vector: np.ndarray
    ) -> float:
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

    def __init__(
        self,
        name: str,
        n_decision_vars: int,
        n_objective_funs: int,
        evaluator: Callable,
    ) -> None:
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

    def evaluate(
        self, decision_vector: np.ndarray, objective_vector: np.ndarray
    ) -> float:
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
            msg = (
                "Decision vector {} is of wrong lenght: "
                "Should be {}, but is {}"
            ).format(
                decision_vector, self.__n_decision_vars, len(decision_vector)
            )
            logger.debug(msg)
            raise ConstraintError(msg)

        if len(objective_vector) != self.__n_objective_funs:
            msg = (
                "Objective vector {} is of wrong lenght:"
                " Should be {}, but is {}"
            ).format(
                objective_vector,
                self.__n_objective_funs,
                len(objective_vector),
            )
            logger.debug(msg)
            raise ConstraintError(msg)

        try:
            result = self.__evaluator(decision_vector, objective_vector)
        except (TypeError, IndexError) as e:
            msg = (
                "Bad arguments {} and {} supllied to the evaluator:" " {}"
            ).format(str(decision_vector), objective_vector, str(e))
            raise ConstraintError(msg)

        return result


# A static variable indicating the supported operators for constructing a
# constraint.
supported_operators: List[str] = ["==", "<", ">"]


def constraint_function_factory(
    lhs: Callable, rhs: float, operator: str
) -> Callable:
    """A function that creates an evaluator to be used with the ScalarConstraint
    class. Constraints should be formulated in a way where all the mathematical
    expression are on the left hand side, and the constants on the right hand
    side.

    Args:
       lhs (Callable): The left hand side of the constraint. Should be a
       callable function representing a mathematical expression.
       rhs (float): The right hand side of a constraint. Represents the right
       hand side of the constraint.
       operator (str): The kind of constraint. Can be '==', '<', '>'.

    Returns
       Callable: A function that can be called to evaluate the rhs and
       which returns representing how the constraint is obeyed. A negative
       value represent a violation of the constraint and a positive value an
       agreement with the constraint. The absolute value of the float is a
       direct indicator how the constraint is violated/agdreed with.

    """
    if operator not in supported_operators:
        msg = "The operator {} supplied is not supported.".format(operator)
        logger.debug(msg)
        raise ValueError(msg)

    if operator == "==":

        def equals(vector: np.ndarray) -> float:
            return -abs(lhs(vector) - rhs)

        return equals

    elif operator == "<":

        def lt(vector: np.ndarray) -> float:
            return rhs - lhs(vector)

        return lt

    elif operator == ">":

        def gt(vector: np.ndarray) -> float:
            return lhs(vector) - rhs

        return gt

    else:
        # if for some reason a bad operator falls through
        msg = "Bad operator argument supplied: {}".format(operator)
        logger.debug(msg)
        raise ValueError(msg)
