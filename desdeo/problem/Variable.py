"""Constaints classes representing variables to be used with the Problem
classes.

"""

import logging
import logging.config
from typing import Tuple
from os import path

import numpy as np

log_conf_path = path.join(path.dirname(path.abspath(__file__)),
                          "../../logger.cfg")
logging.config.fileConfig(fname=log_conf_path,
                          disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class VariableError(Exception):
    """Raised when an error in encountered during the handling of the
    Variable class.

    """
    pass


class Variable:
    """Simple variable with a name, initial value and bounds.

    Args:
        name (str): Name of the variable
        initial_value (float): The initial value of the variable.
        lower_bound (float, optional): Lower bound of the variable. Defaults
            to negative infinity.
        upper_bound (float, optional): Upper of the variable. Defaults
            to positive infinity.

    Attributes:
        name (str): Name of the variable.
        initial_value (float): Initial value of the variable.
        lower_bound (float): Lower bound of the variable.
        upper_bound (float): Upper bound of the variable.

    """

    def __init__(
        self,
        name: str,
        initial_value: float,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
    ) -> None:

        self.__name: str = name
        self.__initial_value: float
        self.__lower_bound: float
        self.__upper_bound: float
        # Check that the bounds make sense
        if not (lower_bound < upper_bound):
            raise VariableError(
                "Lower bound should be less than the upper bound.")

        # Check that the initial value is between the bounds
        if not (lower_bound < initial_value < upper_bound):
            raise VariableError(
                "The initial value should be between the upper and lower "
                "bounds.")

        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__initial_value = initial_value

    @property
    def name(self):
        return self.__name

    @property
    def initial_value(self):
        return self.__initial_value

    def get_bounds(self) -> Tuple[float, float]:
        """Return the bounds of the variables as a tuple.

        Returns:
            tuple(float, float): A tuple consisting of (lower_bound,
            upper_bound)

        """
        return (self.__lower_bound, self.__upper_bound)
