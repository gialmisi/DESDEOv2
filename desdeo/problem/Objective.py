"""Defines Objective classes to be used in Problems

"""

import logging
import logging.config
from abc import ABC, abstractmethod
from os import path
from typing import Callable

import numpy as np

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class ObjectiveError(Exception):
    """Raised when an error related to the Objective class is encountered.

    """


class ObjectiveBase(ABC):
    """The abstract base class for objectives.

    """

    @abstractmethod
    def evaluate(self, decision_vector: np.ndarray) -> float:
        """Evaluates the objective according to a decision variable vector.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
            the evaluation of the objective.

        """
        pass


class ScalarObjective(ObjectiveBase):
    """A simple objective function that returns a scalar.

    Args:
        name (str): Name of the objective.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.

    Attributes:
        name (str): Name of the objective.
        value (float): The current value of the objective function.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.

    Raises:
        ObjectiveError: When ill formed bounds are given.

    """

    def __init__(
        self,
        name: str,
        evaluator: Callable,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
    ) -> None:
        # Check that the bounds make sense
        if not (lower_bound < upper_bound):
            msg = (
                "Lower bound {} should be less than the upper bound " "{}."
            ).format(lower_bound, upper_bound)
            logger.error(msg)
            raise ObjectiveError(msg)

        self.__name: str = name
        self.__evaluator: Callable = evaluator
        self.__value: float = 0.0
        self.__lower_bound: float = lower_bound
        self.__upper_bound: float = upper_bound

    @property
    def name(self) -> str:
        return self.__name

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value: float):
        self.__value = value

    @property
    def evaluator(self) -> Callable:
        return self.__evaluator

    @property
    def lower_bound(self) -> float:
        return self.__lower_bound

    @property
    def upper_bound(self) -> float:
        return self.__upper_bound

    def evaluate(self, decision_vector: np.ndarray) -> float:
        """Evaluate the objective functions value.

        Args:
            variables (np.ndarray): A vector of variables to evaluate the
            objective function with.
        Returns:
            float: The evaluated value of the objective function.

        Raises:
            ObjectiveError: When a bad argument is supplies to the evaluator.

        """
        try:
            result = self.__evaluator(decision_vector)
        except (TypeError, IndexError) as e:
            msg = "Bad argument {} supplied to the evaluator: {}".format(
                str(decision_vector), str(e)
            )
            logger.error(msg)
            raise ObjectiveError(msg)

        # Store the value of the objective
        self.value = result

        return result
