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


class ReferencePointASF(ASFBase):
    """Uses a reference point q and preferenial factors to scalarize a MOO problem.
    Defined in `Miettinen 2010`_ equation (2).

    Arguments:
        preferential_factors (np.ndarray): The preferential factors.
        nadir_point (np.ndarray): The nadir point of the MOO problem to be
        scalarized.
        utopian_point (np.ndarray): The utopian point of the MOO problem to be
        scalarized.
        roo (float): A small number to be used to scale the sm factor in the
        ASF. Defaults to 0.1.

    .. _Miettinen 2010:
        Miettinen, K.; Eskelinen, P.; Ruiz, F. & Luque, M.
        NAUTILUS method: An interactive technique in multiobjective
        optimization based on the nadir point
        Europen Joural of Operational Research, 2010, 206, 426-434

    """

    def __init__(
        self,
        preferential_factors: np.ndarray,
        nadir_point: np.ndarray,
        utopian_point: np.ndarray,
        roo: float = 0.1,
    ):
        self.__preferential_factors = preferential_factors
        self.__nadir_point = nadir_point
        self.__utopian_point = utopian_point
        self.__roo = roo

    @property
    def preferential_factors(self) -> np.ndarray:
        return self.__preferential_factors

    @preferential_factors.setter
    def preferential_factors(self, val: np.ndarray):
        self.__preferential_factors = val

    @property
    def nadir_point(self) -> np.ndarray:
        return self.__nadir_point

    @nadir_point.setter
    def nadir_point(self, val: np.ndarray):
        self.__nadir_point = val

    @property
    def utopian_point(self) -> np.ndarray:
        return self.__utopian_point

    @utopian_point.setter
    def utopian_point(self, val: np.ndarray):
        self.__utopian_point = val

    @property
    def roo(self) -> float:
        return self.__roo

    @roo.setter
    def roo(self, val: float):
        self.__roo = val

    def __call__(
        self, objective_vector: np.ndarray, reference_point: np.ndarray
    ) -> float:
        """The actual implementation of the ASF.

        Arguments:
            objective_vector (np.ndarray): An objective vector calculated using
            some decision variables in the decision space of the MOO probelm.
            reference_point (np.ndarray): Some reference point.

        Returns:
            float: The value of the ASF.

        """
        mu = self.__preferential_factors
        f = objective_vector
        q = reference_point
        roo = self.__roo
        z_nad = self.__nadir_point
        z_uto = self.__utopian_point

        max_term = np.max(mu * (f - q), axis=-1)
        sum_term = roo * np.sum((f - q) / (z_nad - z_uto), axis=-1)

        return max_term + sum_term
