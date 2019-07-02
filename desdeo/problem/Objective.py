"""Defines Objective classes to be used in Problems

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
        __evaluator (Callable): The function to evaluate the objective's value.

    Attributes:
        name (str): Name of the objective.
        value (float): The current value of the objective function.
        evaluator (Callable): The function to evaluate the objective's value.


    Note:
        The evaluator should used named variables. See the examples.

    """
    def __init__(self,
                 name: str,
                 evaluator: Callable) -> None:
        self.__name: str = name
        self.__evaluator: Callable = evaluator
        self.__value: float = 0.0

    @property
    def name(self) -> str:
        return self.__name

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value: float):
        self.__value = value

    def evaluate(self, decision_vector: np.ndarray) -> float:
        """Evaluate the objective functions value.

        Args:
            variables (np.ndarray): A vector of variables to evaluate the
            objective function with.
        Returns:
            float: The evaluated value of the objective function.

        """
        try:
            result = self.__evaluator(decision_vector)
        except (TypeError, IndexError) as e:
            msg = "Bad argument {} supplied to the evaluator: {}".format(
                str(decision_vector), str(e))
            logger.debug(msg)
            raise ObjectiveError(msg)

        # Store the value of the objective
        self.__value = result

        return result
