"""Various interactive multi-objective optimization methods belonging to the
NAUTILUS-family are defined here.

"""

import logging
import logging.config
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from desdeo.methods.InteractiveMethod import (
    InteractiveMethodBase,
    InteractiveMethodError,
)
from desdeo.problem.Problem import ProblemBase
from desdeo.solver.ASF import ReferencePointASF
from desdeo.solver.PointSolver import IdealAndNadirPointSolver
from desdeo.solver.ScalarSolver import (
    ASFScalarSolver,
    EpsilonConstraintScalarSolver,
)

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class Nautilus(InteractiveMethodBase):
    """Implements the basic NAUTILUS methods as presented in `Miettinen 2010`_

    .. _Miettinen 2010:
        Miettinen, K.; Eskelinen, P.; Ruiz, F. & Luque, M.
        NAUTILUS method: An interactive technique in multiobjective
        optimization based on the nadir point
        Europen Joural of Operational Research, 2010, 206, 426-434

    """

    def __init__(self, problem: ProblemBase):
        super().__init__(problem)
        self.__initialization_requirements: List[
            Tuple[str, Type, Callable]
        ] = [
            # The number of iterations the DM wishes to carry out
            (
                "Number of iterations",
                int,
                Nautilus.n_of_iterations.fset,  # type: ignore
            )
        ]
        self.__preference_requirements: List[Tuple[str, Type, Callable]] = [
            # Ranking the objectives by a relative index set
            (
                "Relative importance",
                np.ndarray,
                Nautilus.preference_index_set.fset,  # type: ignore
            ),
            # Ranking the objectives by percentages
            (
                "Percentages",
                np.ndarray,
                Nautilus.preference_percentages.fset,  # type: ignore
            ),
        ]
        # Used to calculate the utopian point from the ideal point
        self.__epsilon: float = 0.0
        self.__n_of_iterations: int = 0  # == itn
        self.__current_iteration: int = 0  # == h
        self.__remaining_iterations: int = 0  # == ith
        self.__lower_bounds: List[np.ndarray] = []
        self.__upper_bounds: List[np.ndarray] = []
        self.__iteration_points: List[np.ndarray] = []  # == z_i^h
        self.__reference_point: np.ndarray = None  # == q
        self.__solutions: List[np.ndarray] = []  # == x^h
        self.__objective_vectors: List[np.ndarray] = []  # == f^h
        self.__distances: List[float] = []  # == d^h
        self.__preference_index_set: np.ndarray = None
        self.__preference_percentages: np.ndarray = None
        self.__preferential_factors: np.ndarray = np.zeros(  # == mu_i^h
            self.problem.n_of_objectives
        )

        self.__scalar_solver: ASFScalarSolver = ASFScalarSolver(self.problem)
        self.__asf: ReferencePointASF = ReferencePointASF(None, None, None)
        self.__scalar_solver.asf = self.__asf

        self.__epsilon_solver: EpsilonConstraintScalarSolver = EpsilonConstraintScalarSolver(
            self.problem
        )

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, val: float):
        self.__epsilon = val

    @property
    def n_of_iterations(self) -> int:
        return self.__n_of_iterations

    @n_of_iterations.setter
    def n_of_iterations(self, val: int):
        """Set the number of total iterations to be carried.

        Arguments:
            val (int): The total number of iterations. Must be positive.

        Raises:
            InteractiveMethodError: val is not positive.

        """
        if val < 0:
            msg = (
                "The number of iterations '{}' "
                "should be positive.".format(str(val))
            )
            logger.debug(msg)
            raise InteractiveMethodError(msg)
        self.__n_of_iterations = val

    @property
    def initialization_requirements(self) -> List[Tuple[str, Type, Callable]]:
        return self.__initialization_requirements

    @property
    def preference_index_set(self) -> np.ndarray:
        return self.__index_set

    @preference_index_set.setter
    def preference_index_set(self, val: np.ndarray):
        """Set the indexes to rank each of the objectives in order of
        importance.

        Arguments:
            val (np.ndarray): A 1D-vector containing the preference index
            corresponding to each obejctive.

        Raise:
            InteractiveMethodError: The length of the index vector does not
            match the number of objectives in the problem.

        """
        if len(val) != self.problem.n_of_objectives:
            msg = (
                "The number of indices '{}' does not match the number "
                "of objectives '{}'"
            ).format(len(val), self.problem.n_of_objectives)
            logger.debug(msg)
            raise InteractiveMethodError(msg)
        elif not (1 <= max(val) <= self.problem.n_of_objectives):
            msg = (
                "The minimum index of importance must be greater or equal "
                "to 1 and the maximum index of improtance must be less "
                "than or equal to the number of objectives in the "
                "problem, which is {}. Check the indices {}"
            ).format(self.problem.n_of_objectives, val)
            logger.debug(msg)
            raise InteractiveMethodError(msg)
        self.__index_set = val

    @property
    def preference_percentages(self) -> np.ndarray:
        return self.__preference_percentages

    @preference_percentages.setter
    def preference_percentages(self, val: np.ndarray):
        """Set the percentages to descripe how to improve each objective.

        Arguments:
            val (np.ndarray): A 1D-vector containing percentages corresponding
            to each objective.

        Raise:
            InteractiveMethod: The lenght of the prcentages vector does not
            match the number of objectives in the problem.

        """
        if len(val) != self.problem.n_of_objectives:
            msg = (
                "The number of percentages '{}' does not match the number "
                "of objectives '{}'"
            ).format(len(val), self.problem.n_of_objectives)
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        elif np.sum(val) != 100:
            msg = (
                "The sum of the percentages must be 100. Current sum" " is {}."
            ).format(np.sum(val))
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        self.__preference_percentages = val

    @property
    def preferential_factors(self) -> np.ndarray:
        return self.__preferential_factors

    @preferential_factors.setter
    def preferential_factors(self, val: np.ndarray):
        self.__preferential_factors = val

    @property
    def scalar_solver(self) -> ASFScalarSolver:
        return self.__scalar_solver

    @scalar_solver.setter
    def scalar_solver(self, val: ASFScalarSolver):
        self.__scalar_solver = val

    @property
    def asf(self) -> ReferencePointASF:
        return self.__asf

    @asf.setter
    def asf(self, val: ReferencePointASF):
        self.__asf = val

    def _parse_parameters(
        self,
        parameters: Dict[str, Any],
        requirements: List[Tuple[str, Type, Callable]],
    ):
        """Parse the given parameters and set them.

        Arguments:
            parameters (Dict[str, Any]): A dict with str keys representing the
            name of the parameter and the value of the paramter.
            requirements (List[Tuple[str, Type, Callable]]): A list containing
            tuples which contain:
                str: The corresponding name of the parameter which works as the
                key in the dictionary supplies as the first parameter.
                Type: The type of paramter.
                Callable: The setter to set the value of the parameter.

        Returns:
            List[str]: A list with the names of the parameters set.

        Raises:
            InteractiveMethodError: When some parameter is missing, has the
            wrong type or has the wrong kind of value.

        """
        parameters_set = []
        for (key, t, setter) in requirements:
            if key in parameters:
                if isinstance(parameters[key], t):
                    # Parameter found and is of correct type
                    setter(self, parameters[key])
                    parameters_set.append(key)
                else:
                    # Wrong type of paramter
                    msg = (
                        "The type '{}' of the supplied initialization "
                        "parameter '{}' does not match the expected "
                        "type'{}'."
                    ).format(str(type(parameters[key])), key, str(t))
                    logger.debug(msg)
                    raise InteractiveMethodError(msg)

        return parameters_set

    def _calculate_iteration_point(self) -> None:
        """Calculate and store a new iteration point.

        """
        # Number of iterations left. The plus one is to ensure the current
        # iteration is also counted.
        ith = self.__remaining_iterations
        z_prev = self.__iteration_points[
            self.__current_iteration - 1
        ]  # == z_h-1
        f_h = self.__objective_vectors[self.__current_iteration]

        z_h = ((ith - 1) / (ith)) * z_prev + (1 / (ith)) * f_h

        self.__iteration_points[self.__current_iteration] = z_h

    def _calculate_distance(self) -> None:
        """Calculate and store the distance to the pareto set for current
        iteration

        """
        self.__distances[self.__current_iteration] = 100 * (
            np.linalg.norm(
                self.__iteration_points[self.__current_iteration]
                - self.problem.nadir
            )
            / (
                np.linalg.norm(
                    self.__objective_vectors[self.__current_iteration]
                    - self.problem.nadir
                )
            )
        )

    def initialize(
        self, itn: int = 5, initialization_parameters: Dict[str, Any] = None
    ) -> np.ndarray:
        """Initialize the method by setting the initialization parameters and
        caluclating the initial bounds of the problem, the nadir and ideal
        point, if not defined in the problem. See initialization_requirements.

        Arguments:
            itn (int): Number of total iterations. Defaults to 5.
            initialization_parameters (Dict[str, Any]): A dict with str keys
            defining the initial parameters for initializing the method.

        Returns:
            np.ndarray: The first iteration point

        Raises:
            InteractiveMethodError: Wrong type of parameter or missing
            parameter in initialization_requirements.

        """
        # If given, parse and set the given initialization parameters
        if initialization_parameters is not None:
            parameters_set = self._parse_parameters(
                initialization_parameters, self.__initialization_requirements
            )
            # Check that all the initialization parameters were set
            for (key, _, _) in self.__initialization_requirements:
                if key not in parameters_set:
                    msg = (
                        "Required initialization parameter " "'{}' not set."
                    ).format(key)
                    logger.debug(msg)
                    raise InteractiveMethodError(msg)
        # Use the named arguments instead
        else:
            self.n_of_iterations = itn

        # Check if the ideal and nadir points are set
        if self.problem.ideal is None and self.problem.nadir is None:
            # Both missing, compute both
            solver = IdealAndNadirPointSolver(self.problem)
            self.problem.ideal, self.problem.nadir = solver.solve()

        elif self.problem.ideal is None:
            # ideal missing, compute it
            solver = IdealAndNadirPointSolver(self.problem)
            self.problem.ideal, _ = solver.solve()

        elif self.problem.nadir is None:
            # nadir missing, compute it
            solver = IdealAndNadirPointSolver(self.problem)
            _, self.problem.nadir = solver.solve()

        self.__current_iteration = 1
        self.__remaining_iterations = self.n_of_iterations

        self.__iteration_points = [None] * self.n_of_iterations
        self.__iteration_points[0] = self.problem.nadir

        self.__upper_bounds = [None] * self.n_of_iterations
        self.__upper_bounds[self.__current_iteration] = self.problem.nadir

        self.__lower_bounds = [None] * self.n_of_iterations
        self.__lower_bounds[self.__current_iteration] = self.problem.ideal

        self.__solutions = [None] * self.n_of_iterations

        self.__objective_vectors = [None] * self.n_of_iterations

        self.__distances = [0.0] * self.n_of_iterations

        self.asf.nadir_point = self.problem.nadir
        self.asf.utopian_point = self.problem.ideal - self.epsilon

        return self.__iteration_points[0]

    def iterate(
        self,
        preference_information: Dict[str, Any] = None,
        index_set: np.ndarray = None,
        percentages: np.ndarray = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """Iterate once according to the user preference.

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]], float]: A tuple
            containing:
                np.ndarray: The current iteration ponint.
                List[Tuple[float, float]]: A list with tuples with the lower
                and upper bounds for the next iteration
                float: The distance of the current iteration point to the
                pareto optimal set.

        Note:
            If both the relative importance and percentages are defined,
            percentages are used.

        """
        if preference_information is not None:
            parameters_set = self._parse_parameters(
                preference_information, self.__preference_requirements
            )
            if len(parameters_set) >= 1:
                used_preference = parameters_set[0]
            else:
                msg = (
                    "The amount of preference parameters set during the "
                    "iteration phase '{}' is too low. Check the preference "
                    "information '{}'"
                ).format(len(parameters_set), preference_information)
                logger.debug(msg)
                raise InteractiveMethodError(msg)

        elif index_set is not None:
            self.preference_index_set = index_set
            used_preference = "Relative importance"

        else:
            if percentages is not None:
                # percentages given
                self.preference_percentages = percentages
            else:
                # percentages not given, use even percentages
                percentages = (
                    100
                    * np.ones(self.problem.n_of_objectives)
                    / self.problem.n_of_objectives
                )
                self.preference_percentages = percentages

            used_preference = "Percentages"

        # Calculate the preferential factors
        if used_preference == "Percentages":
            # use percentages to calcualte the new iteration point
            delta_q = self.preference_percentages / 100
            self.preferential_factors = 1 / (
                delta_q
                * (self.problem.nadir - (self.problem.ideal - self.__epsilon))
            )

        elif used_preference == "Relative importance":
            # Use the relative importance to calcualte the new points
            print(self.preference_index_set)
            for (i, r) in enumerate(self.preference_index_set):
                self.preferential_factors[i] = 1 / (
                    r
                    * (
                        self.problem.nadir[i]
                        - (self.problem.ideal[i] - self.__epsilon)
                    )
                )

        else:
            msg = "Could not compute the preferential factors."
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        # set the current iteration point as the reference point
        self.__reference_point = self.__iteration_points[
            self.__current_iteration - 1
        ]

        # set the preferential factors in the underlaying asf
        self.asf.preferential_factors = self.preferential_factors

        # solve the ASF
        (solution, (objective, _)) = self.scalar_solver.solve(
            self.__reference_point
        )

        # Store the solution and corresponding objective vector
        self.__solutions[self.__current_iteration] = solution
        self.__objective_vectors[self.__current_iteration] = objective[0]

        # calculate a new iteration point
        self._calculate_iteration_point()

        # calculate the lower bounds for the next iteration
        self.__lower_bounds[self.__current_iteration + 1] = np.zeros(
            self.problem.n_of_objectives
        )

        self.__epsilon_solver.epsilons = self.__iteration_points[
            self.__current_iteration
        ]

        for r in range(self.problem.n_of_objectives):
            (_, (objective, _)) = self.__epsilon_solver.solve(r)
            self.__lower_bounds[self.__current_iteration + 1][r] = objective[
                0
            ][r]

        # set the upper bounds
        self.__upper_bounds[
            self.__current_iteration + 1
        ] = self.__iteration_points[self.__current_iteration]

        # Calculate the distance to the pareto optimal set
        self._calculate_distance()

        return (
            self.__iteration_points[self.__current_iteration],
            list(
                zip(
                    self.__lower_bounds[self.__current_iteration + 1],
                    self.__upper_bounds[self.__current_iteration + 1],
                )
            ),
            self.__distances[self.__current_iteration],
        )

    def interact(self, new_remaining_iterations: Optional[int] = None):
        """Change the total number of iterations if supplied and take a step
        backwards if the DM wished to do so.

        """
        if new_remaining_iterations is not None:
            if new_remaining_iterations < self.__remaining_iterations:
                self.__remaining_iterations = new_remaining_iterations
            else:
                msg = (
                    "Bad number of new remaining iterations '{}'. Should "
                    "be less or equal to the current number of remaining "
                    "iterations '{}'."
                ).format(new_remaining_iterations, self.__remaining_iterations)
                logger.debug(msg)
                raise InteractiveMethodError(msg)

        else:
            self.__remaining_iterations -= 1
            self.__current_iteration += 1

            # Handle this in iterate
            # if step_backwards:
            #     # Take a backward step
            #     if short_step:
            #         # take a short step
            #         pass
            #     else:
            #         # iterate and assume to get new preference information
            #         pass

            # else:
            #     # continue next iteration, or stop, if last iteration
            #     pass
