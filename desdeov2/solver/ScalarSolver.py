"""Define different solvers that solve a scalarized version of a MOO problem.

"""
import abc
import logging
import logging.config
from abc import abstractmethod
from os import path
from typing import Any, Optional, Tuple, Union

import numpy as np

from desdeov2.problem.Problem import ProblemBase
from desdeov2.solver.ASF import ASFBase
from desdeov2.solver.NumericalMethods import (
    DiscreteMinimizer,
    NumericalMethodBase,
    ScipyDE,
)

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
        method (NumericalMethodBase): The numerical method to solve the
        scalarizing functions.

    Attributes:
        problem (ProblemBase): The underlaying problem object with the
        specifications of the problem to solve.
        method (NumericalMethodBase): The numerical method to solve the
        scalarizing functions.

    """

    def __init__(
        self, problem: Union[ProblemBase], method: NumericalMethodBase
    ):
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
    def _evaluator(
        self,
        decision_vectors: np.ndarray,
        objective_vectors: Optional[np.ndarray],
    ) -> np.ndarray:
        """A helper function to formulate the scalarized version of the
        underlying MOO problem

        """
        pass

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
        method (NumericalMethodBase): The numerical method to solve the
        scalarizing functions.

    Attributes:
        weights (np.ndarray): The weights corresponsing to each objectie in the
        problem.

    """

    def __init__(self, problem: ProblemBase, method: NumericalMethodBase):
        super().__init__(problem, method)
        self.__weights: np.ndarray = None

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @weights.setter
    def weights(self, val: np.ndarray):
        self.__weights = val

    def _evaluator(
        self,
        decision_vectors: np.ndarray,
        objective_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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
        constraint_vectors: np.ndarray
        if objective_vectors is None:
            objective_vectors, constraint_vectors = self.problem.evaluate(
                decision_vectors
            )
        else:
            constraint_vectors = None

        weighted_sums = np.zeros(len(objective_vectors))
        for ind, elem in enumerate(objective_vectors):
            if constraint_vectors is not None and np.any(
                constraint_vectors[ind] < 0
            ):
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
        bounds = self.problem.get_variable_bounds()

        if isinstance(self.method, DiscreteMinimizer):
            x = self.method.run(
                self._evaluator,
                bounds,
                variables=self.problem.decision_vectors,
                objectives=self.problem.objective_vectors,
            )
        elif isinstance(self.method, ScipyDE):
            # Scipy does not work with vectors, so we just feed it the first
            # value in the vector returned by the evaluator
            func = lambda x, *args: self._evaluator(x)[0]  # noqa: E731
            x = self.method.run(func, bounds)

        return (x, self.problem.evaluate(x))


class EpsilonConstraintScalarSolver(ScalarSolverBase):
    """A class to represent a solver for solving porblems using the epsilon
    constraint method.

    Attributes:
        epsilons (np.ndarray): The epsilon values to set as the upper limit
        for each objective when treated as a constraint.
        to_be_minimized (int): Integer representing which objective function
        should be minimized.

    """

    def __init__(self, problem: ProblemBase, method: NumericalMethodBase):
        super().__init__(problem, method)
        self.__epsilons: np.ndarray
        self.__to_be_minimized: int = 0

    @property
    def epsilons(self) -> np.ndarray:
        return self.__epsilons

    @epsilons.setter
    def epsilons(self, val: np.ndarray):
        """Set the epsilon values

        Args:
            epsilons (np.ndarray): Array of the epsilon values.

        Raises:
            ScalarSolverError: The array with the epsilon values is of the
            wrong length.

        """
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

    def _evaluator(
        self,
        decision_vector: np.ndarray,
        objective_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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

        constraints: np.ndarray
        epsilon_values: np.ndarray

        if objective_vectors is None:
            objective_vectors, constraints = self.problem.evaluate(
                decision_vector
            )
        else:
            constraints = None

        epsilon_values = np.zeros(len(objective_vectors))

        for (ind, elem) in enumerate(objective_vectors):
            if constraints is not None and np.any(constraints[ind] < 0):
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
        bounds = self.problem.get_variable_bounds()

        if isinstance(self.method, DiscreteMinimizer):
            x = self.method.run(
                self._evaluator,
                bounds,
                variables=self.problem.decision_vectors,
                objectives=self.problem.objective_vectors,
            )
        elif isinstance(self.method, ScipyDE):
            # Scipy does not work with vectors, so we just feed it the first
            # value in the vector returned by the evaluator
            func = lambda x, *args: self._evaluator(x)[0]  # noqa: E731
            x = self.method.run(func, bounds)

        return (x, self.problem.evaluate(x))


class ASFScalarSolver(ScalarSolverBase):
    """A class to represent a solver tha uses the achievement scalarizing
    method to solve a multiobjective optimization problem.

    Attributes:
        asf (ASFBase): The ASF to be used in the scalarization of the
        underlying MOO problem.
        reference_point (np.ndarray): The reference point (or similar) that is
        used in the ASF.

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

    def _evaluator(
        self,
        decision_vectors: np.ndarray,
        objective_vectors: Optional[np.ndarray] = None,
        constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """A helper function to express the original problem as an ASF
           problem.

        Args:
            decision_vectors (np.ndarray): An array of arrays representing the
            decision variable values to evaluate the underlaying MOO problem.
            objective_vectors (Optional[np.ndarray]): An array of arrays
            representing the objective vector values of the problem.
            constraints (Optional[np.ndarray]): An array of arrays containing
            the constraints values evaluated with each entry of decision
            vectors and objective vectors

        Returns:
            np.ndarray: An array of the ASF problem values corresponding to
            each decision variable vector.

        Note:
            Requires an achievement scalarizing function to be defined and set
            to the ASFScalarSolver object. See ASF.py for available functions
            and further details.

        """

        asf_values: np.ndarray
        if objective_vectors is None and constraints is None:
            objective_vectors, constraints = self.problem.evaluate(
                decision_vectors
            )
        elif constraints is not None:
            constraints = constraints

        asf_values = np.zeros(len(objective_vectors))  # type: ignore

        for (ind, elem) in enumerate(objective_vectors):  # type: ignore
            if constraints is not None and np.any(constraints[ind] < 0):
                # suicide method for broken constraints
                asf_values[ind] = np.inf
            else:
                asf_values[ind] = self.asf(elem, self.reference_point)

        if np.all(asf_values == np.inf):
            logger.warning(
                "All asf values are inf, result may " "be not sensical."
            )

        return asf_values

    def solve(
        self, reference_point: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Use differential evolution to solve the ASF problem.

        Args:
            reference_point (np.ndarray): An array representing the reference
            point in the objective function space used in the is ASF.
        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: A tuple
            containing the decision variables as the first element and the
            evaluation results of the underlaying problem as the second
            element.

        """
        self.reference_point = reference_point
        bounds = self.problem.get_variable_bounds()

        if isinstance(self.method, DiscreteMinimizer):
            x = self.method.run(
                self._evaluator,
                bounds,
                evaluator_args={
                    "constraints": self.problem.evaluate_constraint_values()
                },
                variables=self.problem.decision_vectors,
                objectives=self.problem.objective_vectors,
            )
        elif isinstance(self.method, ScipyDE):
            # Scipy does not work with vectors, so we just feed it the first
            # value in the vector returned by the evaluator
            func = lambda x, *args: self._evaluator(x)[0]  # noqa: E731
            x = self.method.run(func, bounds)

        return (x, self.problem.evaluate(x))
