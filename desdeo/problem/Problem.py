"""Here various classes are defined that represent multiobjective optimization
problems.

"""

from abc import ABC, abstractmethod


import numpy as np


class ProblemBase(ABC):
    """The base class from which every other class representing a problem should
    derive.  This class presents common interface for message broking in the
    manager.

    """

    @abstractmethod
    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluates the problem using an ensemble of input vectors.

        Args:
            population (np.ndarray): An array of input vectors.

        Returns:
            np.ndarray: An array of solution vectors.

        Note:
            This is an abstrac method.

        """
        pass


class CustomMOProblem(ProblemBase):
    """A multiobjective optimization problem with user defined objective funcitons,
    constraints and variables

    Args:
        TODO

    Returns:
        TODO

    Note:
        WIP

    """
    def __init__(self,
                 objectives: np.ndarray,
                 variables: np.ndarray,
                 constraints: np.ndarray):
        # TODO
        pass
