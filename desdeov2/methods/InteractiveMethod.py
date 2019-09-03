"""Base classes for interactive methods.

"""

import abc
from abc import abstractmethod

from desdeov2.problem.Problem import ProblemBase


class InteractiveMethodError(Exception):
    """Raised when an error is encountered in the InteractiveMethod classes.

    """

    pass


class InteractiveMethodBase(abc.ABC):
    """Defines a common interface to be used by all the interactive MOO
    methods.

    Arguments:
        problem(ProblemBase): The given problem to be solved using an
        interactive method.

    Attributes:
        problem(ProblemBase): The given problem to be solved using an
        interactive method.

    """

    def __init__(self, problem: ProblemBase):
        self.__problem: ProblemBase = problem

    @property
    def problem(self) -> ProblemBase:
        return self.__problem

    @problem.setter
    def problem(self, val: ProblemBase):
        self.__problem = val

    @abstractmethod
    def initialize(self):
        """Initializes the method.

        """
        pass

    @abstractmethod
    def iterate(self):
        """Iterate the method once.

        """
        pass

    @abstractmethod
    def interact(self):
        """Change the state of the method given a decision maker's
        preference.

        """
        pass
