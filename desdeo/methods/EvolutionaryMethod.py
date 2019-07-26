"""Define a family of classes that use evolutionary algorithms to solve MOO
problems for pareto optimal sets"""

import logging
import logging.config
from abc import ABC
from os import path
from typing import List, Optional

import numpy as np

from desdeo.problem.Problem import ProblemBase
from desdeo.utils.frozen import frozen


log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class EvolutionaryError(Exception):
    """Raised when an error related to evolutionary methods is encountered.

    """


class EvolutionaryMethodBase(ABC):
    """A base class that other EA based methods should derive from. Also
    implements commonly used evolutionary algorithms.

    Arguments:
        problem (ProblemBase): The MOO problem to be solved.

    """

    def __init__(self, problem: ProblemBase):
        self.__problem: ProblemBase = problem

    @property
    def problem(self) -> ProblemBase:
        return self.__problem

    @problem.setter
    def problem(self, val: ProblemBase):
        self.__problem = val


@frozen(logger)
class MOEAD(EvolutionaryMethodBase):
    """Implements the algortihm for solving MOO problems using evolutionary
    algorithms based on the decomposition of the problem. Described in `Qingfu
    2007`_

    Arguments:
        problem (ProblemBase): The MOO problem to be solved.

    .. _Qingfu 2007:
        Qingfu, Z. & Hui L.
        MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
        IEEE Transactions on Evolutionary Computation, 2007, Vol. 11, No. 6

    """

    def __init__(self, problem: ProblemBase):
        super().__init__(problem)
        self.__n: int = 0  # The number of subproblems
        self.__lambdas: np.ndarray = None  # The weight vectors
        # Neighborhood size
        self.__t: int = 0
        # Set of non dominated solutions
        self.__epop: List[np.ndarray] = []
        # Poulation of the solutions of each subproblem
        self.__pop: np.ndarray = None
        # Best values found so far for each objective
        self.__z: np.ndarray = None
        # Contains lists of indices representing the weight vector
        # neighborhoods for each weight vector
        self.__b: np.ndarray = None

    @property
    def n(self) -> int:
        return self.__n

    @n.setter
    def n(self, val: int):
        self.__n = val

    @property
    def lambdas(self) -> np.ndarray:
        return self.__lambdas

    @lambdas.setter
    def lambdas(self, val: np.ndarray):
        if len(val) > self.n:
            msg = (
                "The length of the vector containing the weight vectors "
                "must not exceed the number of subproblem considered. "
                "Length '{}', subproblems given '{}'"
            ).format(len(val), self.n)
            logger.debug(msg)
            raise EvolutionaryError(msg)

        elif len(val) == 0:
            msg = "The weight vectors must not be empty."
            logger.debug(msg)
            raise EvolutionaryError(msg)

        if val.shape[1] != self.problem.n_of_objectives:
            msg = (
                "The length of each weight vector '{}' must match the "
                "number of objectives '{}'"
            ).format(val.shape[1], self.problem.n_of_objectives)
            logger.debug(msg)
            raise EvolutionaryError(msg)

        self.__lambdas = val

    @property
    def t(self) -> int:
        return self.__t

    @t.setter
    def t(self, val: int):
        if val > self.n:
            msg = (
                "The number of weight vectors to be considered part of "
                "a neighborhood '{}' cannot exceed the total number of "
                "weight vectors '{}'."
            ).format(val, self.n)
            logger.debug(msg)
            raise EvolutionaryError(msg)
        self.__t = val

    @property
    def epop(self) -> List[np.ndarray]:
        return self.__epop

    @epop.setter
    def epop(self, val: List[np.ndarray]):
        self.__epop = val

    @property
    def pop(self) -> np.ndarray:
        return self.__pop

    @pop.setter
    def pop(self, val: np.ndarray):
        if val.shape != (self.n, self.problem.n_of_variables):
            msg = ("The shape of the population must (number of subproblems, "
                   "number of variables). Given shape '{}'").format(val.shape)
            logger.debug(msg)
            raise EvolutionaryError(msg)
        self.__pop = val

    @property
    def z(self) -> np.ndarray:
        return self.__z

    @z.setter
    def z(self, val: np.ndarray):
        self.__z = val

    @property
    def b(self) -> np.ndarray:
        return self.__b

    @b.setter
    def b(self, val: np.ndarray):
        if val.shape != (self.n, self.t):
            msg = ("The shape of the neighborhoods vector should be (number "
                   "of subproblems, neighborhood size). "
                   "Given shape '{}'").format(val.shape)
            logger.debug(msg)
            raise EvolutionaryError(msg)

        self.__b = val

    def _generate_uniform_set_of_weights(self) -> np.ndarray:
        """Generate a random linear set of weigh vectors uniformly distributed between
        [0, 1).

        """
        return np.random.uniform(size=(self.n, self.problem.n_of_objectives))

    def _compute_neighborhoods(self) -> np.ndarray:
        b = np.zeros((self.n, self.t))
        tmp = np.zeros(self.n)  # hold the distances
        mask = np.full(self.n, True, dtype=bool)  # Mask out the current weight
        for (i, lam) in enumerate(self.lambdas):
            mask[i] = False
            tmp = np.linalg.norm(lam, self.lambdas[mask])
            indices = np.argsort(tmp)[:self.t]
            b[i] = tmp[indices]
            mask[i] = True

        return b

    def initialize(
        self,
        n_subproblems: int,
        neighborhood_size: int,
        weight_vectors: Optional[np.ndarray] = None,
    ):
        """Give the input paramters, initialize variables and generate an
        initial population

        Arguments:
            n_subporblems (int): The number of subproblems considered.
            neighborhood_size (int): The number of weight closest weight
            vectors to be considered part of the neighborhood of each weight
            vector.
            weight_vectors (Optional[np.ndarray]): A vector of size
            n_subproblems containing a uniform spread of weight vectors. If
            None, compute a set of random linearly spread weight vectors.

        """
        self.n = n_subproblems
        self.t = neighborhood_size

        if weight_vectors is not None:
            self.lambdas = weight_vectors
        else:
            self.lambdas = self._generate_uniform_set_of_weights()

        # Compute the distances between each weight vector and define the
        # neighborhoods
        self.b = np.zeros((self.n, self.t))

        # Generate the initial population
        self.pop = np.zeros((self.n, self.problem.n_of_variables))

        # Initialize the best objective vector to just be infinite
        self.z = np.full(self.problem.n_of_objectives, np.inf)
