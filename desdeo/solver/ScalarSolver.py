"""Define different solvers that solve a scalarized version of a MOO problem.

"""
import abc
import logging
import logging.config
from abc import abstractmethod
from os import path
from typing import Any, Callable, Tuple

import numpy as np

from desdeo.problem.Problem import ProblemBase
from desdeo.solver.ASF import ASFBase
from desdeo.solver.NumericalMethods import NumericalMethodBase

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class ScalarSolverError(Exception):
    """Raised when an error related to the solver classes is
    encountered.

    """

    pass


class ScalarSolverBase(abc.ABC):
    """A base class to define the interface for the various solvers that solve
    a scalarized version of the original MOO problem.

    Args:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    """

    def __init__(self, problem: ProblemBase, method: NumericalMethodBase):
        self.__problem = problem
        self.__method = method

    @property
    def problem(self) -> ProblemBase:
        return self.__problem

    @problem.setter
    def problem(self, val: ProblemBase):
        self.__problem = val

    @property
    def method(self) -> NumericalMethodBase:
        return self.__method

    @method.setter
    def method(self, val: NumericalMethodBase):
        self.__method = val

    @abstractmethod
    def solve(
        self, args: Any
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Solves the problem and returns relevant information.

        Args:
            *args(Any): Any kind of arguments that are relevant to the solver.
            See the solvers for further details.

        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: A tuple
            containing the decision variables as the first element and the
            evaluation result of the underlaying porblem as the second element.

        """
        pass


class WeightingMethodScalarSolver(ScalarSolverBase):
    """A class to represent a solver for solving a problems using the
    weighting method.

    Args:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.
        weights (np.ndarray): The weights corresponsing to each objectie in the
        problem.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.
        weights (np.ndarray): The weights corresponsing to each objectie in the
        problem.

    """

    def __init__(self, problem: ProblemBase, method: NumericalMethodBase):
        super().__init__(problem, method)
        self.__weights: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @weights.setter
    def weights(self, val: np.ndarray):
        self.__weights = val

    def _evaluator(self, decision_vectors: np.ndarray) -> np.ndarray:
        """A helper function to transform the output of the MOO problem
        to a weighted sum.

        Args:
            decision_vectors (np.ndarray): An array of arrays representing
            decision variable values to evaluate the MOO problem.

        Returns:
            np.ndarray: An array with the weighted sums corresponding to
            each solution evaluated using the decision variable vectors
            present in decision_vectors.

        Note:
            If any of the constraints are broken for a particular solution, the
            corresponding weighted sum is set to infinity, regardless of the
            weight value. It is therefore assumed that each objective is to be
            minimized.

        """
        objective_vectors: np.ndarray
        constraint_vectors: np.ndarray
        objective_vectors, constraint_vectors = self.problem.evaluate(
            decision_vectors
        )
        weighted_sums = np.zeros(len(objective_vectors))
        for ind, elem in enumerate(objective_vectors):
            if np.any(constraint_vectors[ind] < 0):
                # If any of the constraints is broken, set the sum to
                # infinity
                weighted_sums[ind] = np.inf
            else:
                # Compute the weigted sum for each objective vector using the
                # dot product
                weighted_sums[ind] = np.dot(
                    objective_vectors[ind], self.__weights
                )

        return weighted_sums

    def solve(
        self, weights: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Use differential evolution to solve the weighted sum problem.

        Args:
            weights (np.ndarray): Array of weights to weigh each objective in
            the sum.

        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: A tuple
            containing the decision variables as the first element and the
            evaluation results of the underlaying porblem as the second
            element.

        Note:
            This method might invoke runtime warnings, which are most likely
            related to the usage of infinity (np.inf).

        """
        self.weights = weights

        func = lambda x, *args: self._evaluator(x)[0]  # noqa: E731
        bounds = self.problem.get_variable_bounds()

        # results = differential_evolution(func, bounds, polish=polish)
        x = self.method.run(func, bounds)

        return (x, self.problem.evaluate(x))


class EpsilonConstraintScalarSolver(ScalarSolverBase):
    """A class to represent a solver for solving porblems using the epsilon
    constraint method.

    Args:
        problem (ProblemBase): The underlaying problem obeject with the
        specifications of the problem to solve
        epsilons (np.ndarray): The epsilon values to set as the upper limit
        for each objective when treated as a constraint.

    Attributes:
        problem (ProblemBase): The underlaying problem obeject with the
        specifications of the problem to solve
        epsilons (np.ndarray): The epsilon values to set as the upper limit
        for each objective when treated as a constraint.

    Note:
        To be implemented.

    """

    def __init__(self, problem: ProblemBase, method: NumericalMethodBase):
        super().__init__(problem, method)
        self.__epsilons: np.ndarray
        self.__to_be_minimized: int

    @property
    def epsilons(self) -> np.ndarray:
        return self.__epsilons

    @epsilons.setter
    def epsilons(self, val: np.ndarray):
        if len(val) != self.problem.n_of_objectives:
            msg = (
                "The length of the epsilons array '{}' must match the "
                "number of objectives '{}'."
            ).format(len(val), self.problem.n_of_objectives)
            logger.debug(msg)
            raise ScalarSolverError(msg)

        self.__epsilons = val

    @property
    def to_be_minimized(self) -> int:
        return self.__to_be_minimized

    @to_be_minimized.setter
    def to_be_minimized(self, val: int):
        self.__to_be_minimized = val

    def _evaluator(self, decision_vector: np.ndarray) -> np.ndarray:
        """A helper function to express the original problem as an epsilon
           constraint problem.

        Args:
            decision_vectors (np.ndarray): An array of arrays representing the
            decision variable values to evaluate the underlaying MOO problem.

        Returns:
            np.ndarray: An array of the epsilon constraint problem values
            corresponding to each decision variable vector.

        Note:
            The epsilons attribute  must be set before this function works.

        """

        objective_vectors: np.ndarray
        constraints: np.ndarray
        epsilon_values: np.ndarray

        objective_vectors, constraints = self.problem.evaluate(decision_vector)
        epsilon_values = np.zeros(len(objective_vectors))

        for (ind, elem) in enumerate(objective_vectors):
            if (constraints is not None) and (np.any(constraints[ind] < 0)):
                # suicide method for broken constraints
                epsilon_values[ind] = np.inf
                continue
            else:
                mask = np.array(
                    [i != self.to_be_minimized for i in range(len(elem))]
                )
                if np.any(elem[mask] > self.epsilons[mask]):
                    epsilon_values[ind] = np.inf
                else:
                    epsilon_values[ind] = elem[self.to_be_minimized]

        return epsilon_values

    def solve(self, to_be_minimized: int):
        """Use differential evolution to solve the epsilon constraint problem.

        Args:
            to_be_minimized (int): The index of the problem in the underlaying
            MOProblem to be solved.
        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: A tuple
            containing the decision variables as the first element and the
            evaluation results of the underlaying problem as the second
            element.

        """
        self.to_be_minimized = to_be_minimized

        # Scipy DE solver handles only scalar valued functions
        func = lambda x, *args: self._evaluator(x)[0]  # noqa: E731
        bounds = self.problem.get_variable_bounds()

        x = self.method.run(func, bounds)

        return (x, self.problem.evaluate(x))


class ASFScalarSolver(ScalarSolverBase):
    """A class to represent a solver tha uses the achievement scalarizing
    method to solve a multiobjective optimization problem.

    Args:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.

    """

    def __init__(self, problem: ProblemBase, method: NumericalMethodBase):
        super().__init__(problem, method)
        self.__asf: ASFBase
        self.__reference_point: np.ndarray

    @property
    def asf(self) -> ASFBase:
        return self.__asf

    @asf.setter
    def asf(self, val: ASFBase):
        self.__asf = val

    @property
    def reference_point(self) -> np.ndarray:
        return self.__reference_point

    @reference_point.setter
    def reference_point(self, val: np.ndarray):
        self.__reference_point = val

    def _evaluator(self, decision_vector: np.ndarray) -> np.ndarray:
        """A helper function to express the original problem as an ASF
           problem.

        Args:
            decision_vectors (np.ndarray): An array of arrays representing the
            decision variable values to evaluate the underlaying MOO problem.

        Returns:
            np.ndarray: An array of the ASF problem values corresponding to
            each decision variable vector.

        Note:
            Requires an achievement scalarizing function to be defined and set
            to the ASFScalarSolver object. See ASF.py for available functions
            and further details.

        """

        objective_vectors: np.ndarray
        constraints: np.ndarray
        asf_values: np.ndarray

        objective_vectors, constraints = self.problem.evaluate(decision_vector)
        asf_values = np.zeros(len(objective_vectors))

        for (ind, elem) in enumerate(objective_vectors):
            if (constraints is not None) and (np.any(constraints[ind] < 0)):
                # suicide method for broken constraints
                asf_values[ind] = np.inf
            else:
                asf_values[ind] = self.__asf(elem, self.__reference_point)

        return asf_values

    def solve(
        self, reference_point: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Use differential evolution to solve the ASF problem.

        Args:
            reference_point (np.ndarray): An array representing the reference
            point in the objective function space used the is ASF.
        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: A tuple
            containing the decision variables as the first element and the
            evaluation results of the underlaying problem as the second
            element.

        """
        self.reference_point = reference_point

        func: Callable
        bounds: np.ndarray

        # Scipy DE solver handles only scalar valued functions
        func = lambda x, *args: self._evaluator(x)[0]  # noqa: E731
        bounds = self.problem.get_variable_bounds()

        x = self.method.run(func, bounds)

        return (x, self.problem.evaluate(x))
