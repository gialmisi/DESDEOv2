""" Here different kinds of constraint functions are defined.

"""

from abc import ABC, abstractmethod
from typing import List


class ConstraintBase(ABC):
    """Base class for constaints.

    """

    @abstractmethod
    def evaluate(self) -> bool:
        """Evaluate the constraint functions and return a  bool
        indicating if the constaint has been broken and or not.

        """
        pass


class ScalarConstraint(ConstraintBase):
    """A simple scalar constaint.

    """
    pass
