"""Define a general class and implementations of achievement scalarizing
functions.

"""
import abc
import logging
import logging.config
from abc import abstractmethod
from os import path

import numpy as np

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class ASFError(Exception):
    """Raised when an error related to the ASF classes is encountered.

    """


class ASFBase(abc.ABC):
    """A base class for representing achievement scalarizing functions.
    Instances of the implementations of this class should function as
    function.

    """

    @abstractmethod
    def __call__(
        self, objective_vector: np.ndarray, reference_point: np.ndarray
    ) -> float:
        """Guarantees that every class deriving from this should be usable like
        a function.

        Args:
            objective_vector(np.ndarray): A vector representing the objective
            values of a MOO problem that was solved with some decision
            variables.
            reference_point(np.ndarray): The reference point used in
            calculating the value of the ASF. If an objective's reference value
            is set to np.nan, that objective is ignored in the calculation of
            the ASF.

        Returns:
            float: The result of the ASF function.

        Note:
            The reference point may not always necessarely be feasible, but
            it's dimensions should match that of the objective vector.
        """
        pass


class SimpleASF(ASFBase):
    """Implements a simple order-representing ASF.

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
        """Evaluate the simple order-representing ASF.

        Args:
            objective_vector (np.ndarray): A vector representing a solution in
            the solution space.
            reference_point (np.ndarray): A vector representing a reference
            point in the solution space.

        Note:
            The shaped of objective_vector and reference_point must match.

        """
        if not objective_vector.shape == reference_point.shape:
            msg = (
                "The dimensions of the objective vector {} and "
                "reference_point {} do not match."
            ).format(objective_vector, reference_point)
            logger.debug(msg)
            raise ASFError(msg)

        return np.max(
            np.where(
                np.isnan(reference_point),
                -np.inf,
                self.__weights * (objective_vector - reference_point),
            )
        )
