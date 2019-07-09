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
        decision_vector: np.ndarray,
        reference_point: np.ndarray,
        args: Optional[Any],
    ) -> float:
        """Guarantees that every class deriving from this should be usable like
        a function.

        Args:
            decision_vector(np.ndarray): A vector representing decision
            variables.
            reference_point(np.ndarray): The reference point used in
            calculating the value of the ASF.

        Returns:
            float: The result of the ASF function.

        Note:
            The reference point may not necessarely be feasible, but it's
            dimension should match that of the decision_vector.
        """
        pass
