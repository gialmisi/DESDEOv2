"""Define a family of classes that use evolutionary algorithms to solve MOO
problems for pareto optimal sets"""

import logging
import logging.config
from abc import ABC
from os import path
from typing import List, Optional, Tuple

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
        # Corresponding colution vectors to epop
        self.__epop_fs: List[np.ndarray] = []     
        # Poulation of the solutions of each subproblem
        self.__pop: np.ndarray = None
        # The objective vectors corresponsing to each subproblem
        self.__fs: np.ndarray = None
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
        """Set the weight vectors.

        Arguments:
            val (np.ndarray): Vector of weight vectors

        Raises:
            EvolutionaryError: Incorrect dimensions of vector of weights.
        """
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
        """Set the size of the neighborhoods.

        Arguments:
            val (int): The size of the neighborhoods.

        Raises:
            EvolutionaryError: val is bigger that the total number of weight
            vectors, or it is negative.

        """
        if val > self.n:
            msg = (
                "The number of weight vectors to be considered part of "
                "a neighborhood '{}' cannot exceed the total number of "
                "weight vectors '{}'."
            ).format(val, self.n)
            logger.debug(msg)
            raise EvolutionaryError(msg)

        elif val < 0:
            msg = "The neighborhood size must be positive."
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
    def epop_fs(self) -> List[np.ndarray]:
        return self.__epop_fs

    @epop_fs.setter
    def epop_fs(self, val: List[np.ndarray]):
        self.__epop_fs = val

    @property
    def pop(self) -> np.ndarray:
        return self.__pop

    @pop.setter
    def pop(self, val: np.ndarray):
        """Set the population of solution vectors.

        Arguments:
            val (np.ndarray): Vector containing solutions to the underlying MOO
            problem.

        Raises:
            EvolutionaryError: Incorrect shape of val.

        """
        if val.shape != (self.n, self.problem.n_of_variables):
            msg = (
                "The shape of the population must (number of subproblems, "
                "number of variables). Given shape '{}'"
            ).format(val.shape)
            logger.debug(msg)
            raise EvolutionaryError(msg)
        self.__pop = val

    @property
    def fs(self) -> np.ndarray:
        return self.__fs

    @fs.setter
    def fs(self, val: np.ndarray):
        """Set the vector of objectives that correspond to the solution in the
        population.

        Arguments:
            val (np.ndarray): A vector of objective vectors.

        Raises:
            EvolutionaryError: Incorrect shape of val.

        """
        if val.shape != (self.n, self.problem.n_of_objectives):
            msg = (
                "The shape of the vector containing the objective "
                "vectors must (number of subproblems, "
                "number of objectives). Given shape '{}'"
            ).format(val.shape)
            logger.debug(msg)
            raise EvolutionaryError(msg)
        self.__fs = val

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
        """Set the vector of neighborhood vectors.

        Arguments:
            val (np.ndarray): A vector of ints representing the neighborhood of
            each weight vector.

        Raises:
            EvolutionaryError: Incorrect shape of val.

        """
        if val.shape != (self.n, self.t):
            msg = (
                "The shape of the neighborhoods vector should be (number "
                "of subproblems, neighborhood size). "
                "Given shape '{}'"
            ).format(val.shape)
            logger.debug(msg)
            raise EvolutionaryError(msg)

        self.__b = val

    def _generate_uniform_set_of_weights(self) -> np.ndarray:
        """Generate a random linear set of weigh vectors uniformly distributed between
        [0, 1).

        Returns:
            np.ndarray: A vector containing weight vectors distributed
            uniformly.

        """
        return np.random.uniform(size=(self.n, self.problem.n_of_objectives))

    def _compute_neighborhoods(self) -> np.ndarray:
        """Compute the neighborhoods for each weight vector. A neighborhood is
        defined as a vector of unique indices, each index repsesenting a weight
        vector in the neighborhood.

        Returns:
            np.ndarray: A vector of the neighborhoods

        """
        b = np.zeros((self.n, self.t), dtype=int)
        for (i, lam) in enumerate(self.lambdas):
            indices = np.argsort(np.linalg.norm(lam - self.lambdas, axis=1))[
                1 : self.t + 1
            ]
            b[i] = indices

        return b

    def _generate_feasible_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates an initial feasible population of solution vectors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the solution
            vectors and the corresponding objevtive vectors.

        Note:
            This function will try to retry the generation of each solution
            until it finds a feasible one. This could results in
            complication. Beware!

        """
        n_vars = self.problem.n_of_variables
        pop = np.zeros((self.n, n_vars))
        fs = np.zeros((self.n, self.problem.n_of_objectives))
        lows = self.problem.get_variable_lower_bounds()
        highs = self.problem.get_variable_upper_bounds()

        for i in range(len(pop)):
            while True:
                x = np.random.uniform(lows, highs, (1, n_vars))
                f, cons = self.problem.evaluate(x)
                if cons is None or np.all(cons >= 0):
                    pop[i] = x
                    fs[i] = f
                    break

        return pop, fs

    def initialize(
        self,
        n_subproblems: int,
        neighborhood_size: int,
        weight_vectors: Optional[np.ndarray] = None,
        initial_population: Optional[np.ndarray] = None,
    ):
        """Give the input parameters, initialize variables and generate an
        initial population

        Arguments:
            n_subporblems (int): The number of subproblems considered.
            neighborhood_size (int): The number of weight closest weight
            vectors to be considered part of the neighborhood of each weight
            vector.
            weight_vectors (Optional[np.ndarray]): A vector of size
            n_subproblems containing a uniform spread of weight vectors. If
            None, compute a set of random linearly spread weight vectors.
            initial_population (np.ndarray): A vector of initial solution
            vectors for the problem.

        """
        self.n = n_subproblems
        self.t = neighborhood_size

        if weight_vectors is not None:
            self.lambdas = weight_vectors
        else:
            self.lambdas = self._generate_uniform_set_of_weights()

        # Compute the distances between each weight vector and define the
        # neighborhoods
        self.b = self._compute_neighborhoods()

        # Generate the initial population
        if initial_population is None:
            self.pop, self.fs = self._generate_feasible_population()
        else:
            self.pop = initial_population
            self.fs, _ = self.problem.evaluate(self.pop)

        # Initialize the best objective values
        self.z = np.min(self.fs, axis=0)

    def run(self):
        for i in range(self.n):
            k, j = np.random.choice(self.b[i], size=2)

            while True:
                # make a random amalgamation of the two solutions until a
                # feasible one is born
                w_k = np.random.uniform(0.0, 1.0, size=self.pop[k].shape)
                w_j = np.random.uniform(0.0, 1.0, size=self.pop[j].shape)
                y = (w_k * self.pop[k] + w_j * self.pop[j])

                y_objectives, y_consts = self.problem.evaluate(y)
                if y_consts is None:
                    break
                elif np.all(y_consts >= 0):
                    # solution is feasible, continue
                    break

            # update best objective values found so far
            for m, elem in enumerate(self.z):
                if y_objectives[0][m] < elem:
                    self.z[m] = y_objectives[0][m]

            # update neighboring solutions
            for h in self.b[i]:
                y_te = np.max(self.lambdas[h] * np.abs(y_objectives[0] - self.z))
                x_te = np.max(self.lambdas[h] * np.abs(self.fs[h] - self.z))

                if y_te <= x_te:
                    self.pop[h] = y
                    self.fs[h] = y_objectives[0]

            # update the external population
            dominated = False
            for ind, sol in enumerate(self.epop):
                # if y dominates any member in EP, remove the member
                if np.all(y_objectives[0] <= sol) and np.any(y_objectives[0] < sol):
                    print(y_objectives[0], "dominates", sol)
                    self.epop.pop(ind)
                    self.epop_fs.pop(ind)
                    continue
                # check if any member in the population dominates y
                elif np.all(sol <= y_objectives[0]) and np.any(
                    sol < y_objectives[0]
                ):
                    dominated = True

            # if y is not dominated, add it to the EP
            if not dominated:
                self.epop.append(y_objectives[0])
                self.epop_fs.append(y)
