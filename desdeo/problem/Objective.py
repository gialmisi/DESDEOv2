"""Defines Objective classes to be used in Problems

"""

import logging
import logging.config
from os import path
from abc import ABC, abstractmethod
from typing import List, Callable, Dict

from desdeo.problem.Variable import Variable

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
    def evaluate(self, variables: List[Variable]) -> float:
        """Evaluates the objective according to a decision variable vector.

        Args:
            variables (List[Variable]): A vector of Variables to be used in
            the evaluation of the objective.

        """
        pass


class ScalarObjective(ObjectiveBase):
    """A simple objective function that returns a scalar.

    Args:
        name (str): Name of the objective.
        __evaluator (Callable): The function to evaluate the objective's value.
        __variable_names (List[str]): A list of the variable names present in
        the objective's callable function.
    Attributes:
        name (str): Name of the objective.
        __evaluator (Callable): The function to evaluate the objective's value.
        __variable_names (List[str]): A list of the variable names present in
        the objective's callable function.
        value (float): The current value of the objective function.

    Note:
        The evaluator should used named variables. See the examples.

    """
    def __init__(self,
                 name: str,
                 evaluator: Callable,
                 variable_names: List[str]) -> None:
        self.__name: str = name
        self.__evaluator: Callable = evaluator
        self.__variable_names: List[str] = variable_names
        self.__value: float = 0.0

    @property
    def name(self) -> str:
        return self.__name

    @property
    def variable_names(self) -> List[str]:
        return self.__variable_names

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value: float):
        self.__value = value

    def evaluate(self, variables: List[Variable]) -> float:
        """Evaluate the objective functions value.

        Args:
            variables (List[Variable]): A vector of variables to evaluate the
            objective function with.
        Returns:
            float: The evaluated value of the objective function.

        """
        arguments = {var.name: var.current_value for var in variables}

        try:
            result = self.__evaluator(**arguments)
        except TypeError:
            msg = "Bad arguments {} supplied to the evaluator.".format(
                str(arguments))
            logger.debug(msg)
            raise ObjectiveError(msg)

        # Store the value of the objective
        self.value = result

        return result
