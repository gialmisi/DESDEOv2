"""Various interactive multi-objective optimization methods belonging to the
NAUTILUS-family are defined here.

"""

import logging
import logging.config
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
                Nautilus.itn.fset,  # type: ignore
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
        self.__itn: int = 0  # total number of iterations
        self.__h: int = 0  # current iteration
        self.__ith: int = 0  # number of remaining iterations
        self.__lower_bounds: List[np.ndarray] = []
        self.__upper_bounds: List[np.ndarray] = []
        self.__zs: List[np.ndarray] = []  # iteration points
        self.__q: np.ndarray = None  # The current reference point
        self.__xs: List[np.ndarray] = []  # solutions for each iteration
        self.__fs: List[np.ndarray] = []  # objectives for each iteration
        self.__ds: List[float] = []  # distances for each iteration
        self.__preference_index_set: np.ndarray = None
        self.__preference_percentages: np.ndarray = None
        self.__mus: np.ndarray = np.zeros(  # current preferential factors
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
    def itn(self) -> int:
        return self.__itn

    @itn.setter
    def itn(self, val: int):
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
        self.__itn = val

    @property
    def h(self) -> int:
        return self.__h

    @h.setter
    def h(self, val: int):
        self.__h = val

    @property
    def ith(self) -> int:
        return self.__ith

    @ith.setter
    def ith(self, val: int):
        self.__ith = val

    @property
    def zs(self) -> List[np.ndarray]:
        return self.__zs

    @zs.setter
    def zs(self, val: List[np.ndarray]):
        self.__zs = val

    @property
    def q(self) -> np.ndarray:
        return self.__q

    @q.setter
    def q(self, val: np.ndarray):
        self.__q = val

    @property
    def xs(self) -> List[np.ndarray]:
        return self.__xs

    @xs.setter
    def xs(self, val: List[np.ndarray]):
        self.__xs = val

    @property
    def fs(self) -> List[np.ndarray]:
        return self.__fs

    @fs.setter
    def fs(self, val: List[np.ndarray]):
        self.__fs = val

    @property
    def ds(self) -> List[float]:
        return self.__ds

    @ds.setter
    def ds(self, val: List[float]):
        self.__ds = val

    @property
    def mus(self) -> np.ndarray:
        return self.__mus

    @mus.setter
    def mus(self, val: np.ndarray):
        self.__mus = val

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
        return self.mus

    @preferential_factors.setter
    def preferential_factors(self, val: np.ndarray):
        self.mus = val

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
        ith = self.ith
        z_prev = self.zs[self.h - 1]  # == z_h-1
        f_h = self.fs[self.h]

        z_h = ((ith - 1) / (ith)) * z_prev + (1 / (ith)) * f_h

        self.zs[self.h] = z_h

    def _calculate_distance(self) -> None:
        """Calculate and store the distance to the pareto set for current
        iteration

        """
        self.ds[self.h] = 100 * (
            np.linalg.norm(self.zs[self.h] - self.problem.nadir)
            / (np.linalg.norm(self.fs[self.h] - self.problem.nadir))
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
            self.itn = itn

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

        self.h = 1
        self.ith = self.itn

        self.zs = [None] * self.itn
        self.zs[0] = self.problem.nadir

        self.__upper_bounds = [None] * self.itn
        self.__upper_bounds[self.h] = self.problem.nadir

        self.__lower_bounds = [None] * self.itn
        self.__lower_bounds[self.h] = self.problem.ideal

        self.xs = [None] * self.itn

        self.fs = [None] * self.itn

        self.ds = [0.0] * self.itn

        self.asf.nadir_point = self.problem.nadir
        self.asf.utopian_point = self.problem.ideal - self.epsilon

        return self.zs[0]

    def iterate(
        self,
        preference_information: Dict[str, Any] = None,
        index_set: np.ndarray = None,
        percentages: np.ndarray = None,
        short_step: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """Iterate once according to the user preference.

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]], float]: A tuple
            containing:
                np.ndarray: The current iteration point.
                List[Tuple[float, float]]: A list with tuples with the lower
                and upper bounds for the next iteration
                float: The distance of the current iteration point to the
                pareto optimal set.

        Note:
            If both the relative importance and percentages are defined,
            percentages are used.

        """
        if short_step:
            if self.h == 1:
                msg = "Cannot take a backwards step on the first iteration."
                logger.debug(msg)
                raise InteractiveMethodError(msg)

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

        elif not short_step:
            # no short step, use new preference information
            if index_set is not None:
                self.preference_index_set = index_set
                used_preference = "Relative importance"

            elif percentages is not None:
                # percentages given
                self.preference_percentages = percentages
                used_preference = "Percentages"

            else:
                # both preferences not given, use from previus step
                used_preference = "Previous"

        # Calculate the preferential factors
        if not short_step:
            if used_preference == "Percentages":
                # use percentages to calcualte the new iteration point
                delta_q = self.preference_percentages / 100
                self.preferential_factors = 1 / (
                    delta_q
                    * (
                        self.problem.nadir
                        - (self.problem.ideal - self.__epsilon)
                    )
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

            elif used_preference == "Previous":
                if self.h == 1:
                    # First iteration, cannot use previous preference
                    msg = (
                        "Cannot use previous' step preference on the first "
                        "iteration."
                    )
                    logger.debug(msg)
                    raise InteractiveMethodError(msg)
                pass

            else:
                msg = "Could not compute the preferential factors."
                logger.debug(msg)
                raise InteractiveMethodError(msg)

        if not short_step and used_preference != "Previous":
            # No need to calculate all this if short step
            # Take a normal step
            # set the current iteration point as the reference point
            self.q = self.zs[self.h - 1]

            # set the preferential factors in the underlaying asf
            self.asf.preferential_factors = self.preferential_factors

            # solve the ASF
            (solution, (objective, _)) = self.scalar_solver.solve(self.q)

            # Store the solution and corresponding objective vector
            self.xs[self.h] = solution
            self.fs[self.h] = objective[0]

            # calculate a new iteration point
            self._calculate_iteration_point()

        elif used_preference == "Previous":
            # Use the solution and objective of the last step
            self.xs[self.h] = self.xs[self.h - 1]
            self.fs[self.h] = self.fs[self.h - 1]

        else:
            # Take a short step
            # Update the current iteration point
            self.zs[self.h] = 0.5 * self.zs[self.h] + 0.5 * self.zs[self.h - 1]

        # calculate the lower bounds for the next iteration
        self.__lower_bounds[self.h + 1] = np.zeros(
            self.problem.n_of_objectives
        )

        self.__epsilon_solver.epsilons = self.zs[self.h]

        for r in range(self.problem.n_of_objectives):
            (_, (objective, _)) = self.__epsilon_solver.solve(r)
            self.__lower_bounds[self.h + 1][r] = objective[0][r]

            # set the upper bounds
            self.__upper_bounds[self.h + 1] = self.zs[self.h]

        # Calculate the distance to the pareto optimal set
        self._calculate_distance()

        return (
            self.zs[self.h],
            list(
                zip(
                    self.__lower_bounds[self.h + 1],
                    self.__upper_bounds[self.h + 1],
                )
            ),
            self.ds[self.h],
        )

    def interact(
        self,
        new_remaining_iterations: Optional[int] = None,
        step_back: bool = False,
    ) -> Union[int, Tuple[np.ndarray, np.ndarray]]:
        """Change the total number of iterations if supplied and take a step
        backwards if the DM wished to do so.

        """
        if new_remaining_iterations is not None:
            if new_remaining_iterations < self.ith:
                self.ith = new_remaining_iterations
            else:
                msg = (
                    "Bad number of new remaining iterations '{}'. Should "
                    "be less or equal to the current number of remaining "
                    "iterations '{}'."
                ).format(new_remaining_iterations, self.ith)
                logger.debug(msg)
                raise InteractiveMethodError(msg)

        if not step_back:
            # Advnce the current iteration
            self.ith -= 1
            self.h += 1

            if self.ith == 1:
                # Terminate and return the solution
                return (self.xs[self.h], self.fs[self.h])

        else:
            # Taking a step backwards, do nothing
            pass

        return self.ith
