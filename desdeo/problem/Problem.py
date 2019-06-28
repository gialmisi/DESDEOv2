"""Here various classes are defined that represent multiobjective optimization
problems.

"""

from abc import ABC, abstractmethod
from typing import List


import numpy as np
from desdeo.problem.Variable import Variable
from desdeo.problem.Objective import ScalarObjective
from desdeo.problem.Constraint import ScalarConstraint


class ProblemBase(ABC):
    """The base class from which every other class representing a problem should
    derive.  This class presents common interface for message broking in for
    the manager.

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


class ScalarMOProblem(ProblemBase):
    """A multiobjective optimization problem with user defined objective funcitons,
    constraints and variables. The objectives each return a single scalar.

    Args:
        TODO

    Returns:
        TODO

    Note:
        WIP

    """
    def __init__(self,
                 objectives: List[ScalarObjective],
                 variables: List[Variable],
                 constraints: List[ScalarConstraint]) -> None:
        # TODO
        pass
