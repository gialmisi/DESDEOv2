"""Define a general class and implementations of achievement scalarizing
functions.

"""
import abc
from abc import abstractmethod
from typing import Any, Optional

import numpy as np


class ASFBase(abc.ABC):
    """A base class for representing achievement scalarizing functions.
    Instances of the implementations of this class should function as
    function.

    """

    @abstractmethod
    def __call__(
        self,
        objective_vector: np.ndarray,
        reference_point: np.ndarray,
        args: Optional[Any],
    ) -> float:
        """Guarantees that every class deriving from this should be usable like
        a function.

        Args:
            objective_vector(np.ndarray): A vector representing the objective
            values of a MOO problem that was solved with some decision
            variables.
            reference_point(np.ndarray): The reference point used in
            calculating the value of the ASF.

        Returns:
            float: The result of the ASF function.

        Note:
            The reference point may not always necessarely be feasible, but it's
            dimensions should match that of the objective vector.
        """
        pass


class SimpleASF(ASFBase):
    """Implements a simple order-representing ASF.
    TODO: SOURCE!

    Args:
        weights (np.ndarray): A weight vector that holds weights. It's
        length should match the number of objectives in the underlying
        MOO problem the achievement problem aims to solve.

    Attributes:
        weights (np.ndarray): A weight vector that holds weights. It's
        length should match the number of objectives in the underlying
        MOO problem the achievement problem aims to solve.

    """

    def __init__(self, weights: np.ndarray):
        self.__weights = weights

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @weights.setter
    def weights(self, val: np.ndarray):
        self.__weights = val

    def __call__(
        self, objective_vector: np.ndarray, reference_point: np.ndarray
    ) -> float:
        return np.max(self.__weights * (objective_vector - reference_point))
