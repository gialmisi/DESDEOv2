"""Define a general class and implementations of achievement scalarizing
functions.

"""
import abc
import logging
import logging.config
from abc import abstractmethod
from os import path
from typing import List, Union


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
    ) -> Union[float, np.ndarray]:
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
    ) -> Union[float, np.ndarray]:
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
                self.weights * (objective_vector - reference_point),
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
    ) -> Union[float, np.ndarray]:
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


class MaxOfTwoASF(ASFBase):
    """Implements the ASF as defined in eq. 3.1 `Miettinen 2006`_

    .. _Miettinen 2006:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922

    """

    def __init__(
        self,
        nadir: np.ndarray,
        ideal: np.ndarray,
        lt_inds: List[int],
        lte_inds: List[int],
        rho: float = 1e-6,
        rho_sum: float = 1e-6,
    ):
        self.__nadir = nadir
        self.__ideal = ideal
        self.__lt_inds = lt_inds
        self.__lte_inds = lte_inds
        self.__rho = rho
        self.__rho_sum = rho_sum

    @property
    def nadir(self) -> np.ndarray:
        return self.__nadir

    @nadir.setter
    def nadir(self, val: np.ndarray):
        self.__nadir = val

    @property
    def ideal(self) -> np.ndarray:
        return self.__ideal

    @ideal.setter
    def ideal(self, val: np.ndarray):
        self.__ideal = val

    @property
    def lt_inds(self) -> List[int]:
        return self.__lt_inds

    @lt_inds.setter
    def lt_inds(self, val: List[int]):
        self.__lt_inds = val

    @property
    def lte_inds(self) -> List[int]:
        return self.__lte_inds

    @lte_inds.setter
    def lte_inds(self, val: List[int]):
        self.__lte_inds = val

    @property
    def rho(self) -> float:
        return self.__rho

    @rho.setter
    def rho(self, val: float):
        self.__rho = val

    @property
    def rho_sum(self) -> float:
        return self.__rho_sum

    @rho_sum.setter
    def rho_sum(self, val: float):
        self.__rho_sum = val

    def __call__(
        self, objective_vector: np.ndarray, reference_point: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Compute the value for the ASF using the objective vectors and
        reference point given.

        Parameters:
            objective_vector(np.ndarray): The objective vector(s)
            reference_point(np.ndarray): The reference point

        Returns:
            Union[float, np.ndarray]: The asf value(s). Either a float for if a
            single objective vector was given, or an array if multiple
            objective vectors were given.

        """
        # assure this function works with single objective vectors
        if objective_vector.ndim == 1:
            f = objective_vector.reshape((1, -1))
        else:
            f = objective_vector

        ii = self.lt_inds
        jj = self.lte_inds
        z = reference_point
        nad = self.nadir
        ide = self.ideal
        uto = self.ideal - self.rho

        lt_term = (f[:, ii] - ide[ii]) / (nad[ii] - uto[ii])
        lte_term = (f[:, jj] - z[jj]) / (nad[jj] - uto[jj])
        max_term = np.max(np.hstack((lt_term, lte_term)), axis=1)
        sum_term = self.rho_sum * np.sum(f / (nad - uto), axis=1)

        return max_term + sum_term


class StomASF(ASFBase):
    """Implementation of the satisficing trade-off method (STOM) as presented
    in `Miettinen 2006` equation (3.2)

    .. _Miettinen 2006:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922

    """

    def __init__(
        self, ideal: np.ndarray, rho: float = 1e-6, rho_sum: float = 1e-6
    ):
        self.__ideal = ideal
        self.__rho = rho
        self.__rho_sum = rho_sum

    @property
    def ideal(self) -> np.ndarray:
        return self.__ideal

    @ideal.setter
    def ideal(self, val: np.ndarray):
        self.__ideal = val

    @property
    def rho(self) -> float:
        return self.__rho

    @rho.setter
    def rho(self, val: float):
        self.__rho = val

    @property
    def rho_sum(self) -> float:
        return self.__rho_sum

    @rho_sum.setter
    def rho_sum(self, val: float):
        self.__rho_sum = val

    def __call__(
        self, objective_vectors: np.ndarray, reference_point: np.ndarray
    ):
        # assure this function works with single objective vectors
        if objective_vectors.ndim == 1:
            f = objective_vectors.reshape((1, -1))
        else:
            f = objective_vectors

        z = reference_point
        uto = self.ideal - self.rho

        max_term = np.max((f - uto) / (z - uto), axis=1)
        sum_term = self.rho_sum * np.sum((f) / (z - uto), axis=1)

        return max_term + sum_term


class PointMethodASF(ASFBase):
    """Implementation of the reference point based ASF as presented
    in `Miettinen 2006` equation (3.3)

    .. _Miettinen 2006:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922

    Note:
        Lack of better name...

    """

    def __init__(
        self,
        nadir: np.ndarray,
        ideal: np.ndarray,
        rho: float = 1e-6,
        rho_sum: float = 1e-6,
    ):
        self.__nadir = nadir
        self.__ideal = ideal
        self.__rho = rho
        self.__rho_sum = rho

    @property
    def nadir(self) -> np.ndarray:
        return self.__nadir

    @nadir.setter
    def nadir(self, val: np.ndarray):
        self.__nadir = val

    @property
    def ideal(self) -> np.ndarray:
        return self.__ideal

    @ideal.setter
    def ideal(self, val: np.ndarray):
        self.__ideal = val

    @property
    def rho(self) -> float:
        return self.__rho

    @rho.setter
    def rho(self, val: float):
        self.__rho = val

    @property
    def rho_sum(self) -> float:
        return self.__rho_sum

    @rho_sum.setter
    def rho_sum(self, val: float):
        self.__rho_sum = val

    def __call__(
        self, objective_vectors: np.ndarray, reference_point: np.ndarray
    ):
        # assure this function works with single objective vectors
        if objective_vectors.ndim == 1:
            f = objective_vectors.reshape((1, -1))
        else:
            f = objective_vectors

        z = reference_point
        nad = self.nadir
        uto = self.ideal - self.rho

        max_term = np.max((f - z) / (nad - uto), axis=1)
        sum_term = self.rho_sum * np.sum((f) / (nad - uto), axis=1)

        return max_term + sum_term


class AugmentedGuessASF(ASFBase):
    """Implementation of the augmented GUESS related ASF as presented in
    `Miettinen 2006` equation (3.4)

    .. _Miettinen 2006:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922

    """

    def __init__(
        self,
        nadir: np.ndarray,
        indx_to_exclude: List[int],
        rho_sum: float = 1e-6,
    ):
        self.__nadir = nadir
        self.__indx_to_exclude = indx_to_exclude
        self.__rho_sum = rho_sum

    @property
    def nadir(self) -> np.ndarray:
        return self.__nadir

    @nadir.setter
    def nadir(self, val: np.ndarray):
        self.__nadir = val

    @property
    def indx_to_exclude(self) -> List[int]:
        return self.__indx_to_exclude

    @indx_to_exclude.setter
    def indx_to_exclude(self, val: List[int]):
        self.__indx_to_exclude = val

    @property
    def rho_sum(self) -> float:
        return self.__rho_sum

    @rho_sum.setter
    def rho_sum(self, val: float):
        self.__rho_sum = val

    def __call__(
        self, objective_vectors: np.ndarray, reference_point: np.ndarray
    ):
        # assure this function works with single objective vectors
        if objective_vectors.ndim == 1:
            f = objective_vectors.reshape((1, -1))
        else:
            f = objective_vectors

        z = reference_point
        nad = self.nadir
        ex_mask = np.full((f.shape[1]), True, dtype=bool)
        ex_mask[self.indx_to_exclude] = False

        max_term = np.max(
            (f[:, ex_mask] - nad[ex_mask]) / (nad[ex_mask] - z[ex_mask]),
            axis=1,
        )
        sum_term = np.sum((f) / (nad - z), axis=1)

        return max_term + sum_term
