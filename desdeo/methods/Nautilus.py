"""Various interactive multi-objective optimization methods belonging to the
NAUTILUS-family are defined here.

"""

import logging
import logging.config
from os import path
from typing import List, Optional, Tuple, Union

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
        self.__mu: np.ndarray = np.zeros(  # current preferential factors
            self.problem.n_of_objectives
        )

        self.__scalar_solver: ASFScalarSolver = ASFScalarSolver(self.problem)
        self.__asf: ReferencePointASF = ReferencePointASF(None, None, None)
        self.__scalar_solver.asf = self.__asf

        self.__epsilon_solver: EpsilonConstraintScalarSolver = EpsilonConstraintScalarSolver(
            self.problem
        )

        # flags for the iteration phase
        self.__use_previous_preference: bool = False
        self.__step_back: bool = False
        self.__short_step: bool = False
        self.__first_iteration: bool = True

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
                "The number of total iterations "
                "should be positive. Given iterations '{}'".format(str(val))
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
        if val < 0:
            msg = (
                "The given number of iterations left "
                "should be positive. Given iterations '{}'".format(str(val))
            )
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        elif val > self.__ith:
            msg = (
                "The given number of iterations '{}' left should be less "
                "than the current number of iterations left '{}'"
            ).format(val, self.__ith)
            logger.debug(msg)
            raise InteractiveMethodError(msg)
        self.__ith = val

    @property
    def lower_bounds(self) -> List[np.ndarray]:
        return self.__lower_bounds

    @lower_bounds.setter
    def lower_bounds(self, val: List[np.ndarray]):
        self.__lower_bounds = val

    @property
    def upper_bounds(self) -> List[np.ndarray]:
        return self.__upper_bounds

    @upper_bounds.setter
    def upper_bounds(self, val: List[np.ndarray]):
        self.__upper_bounds = val

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
    def mu(self) -> np.ndarray:
        return self.__mu

    @mu.setter
    def mu(self, val: np.ndarray):
        self.__mu = val

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
                "The number of given percentages '{}' does not match the "
                "number of objectives '{}'"
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

    def _calculate_iteration_point(self) -> np.ndarray:
        """Calculate and store a new iteration point.

        """
        # Number of iterations left. The plus one is to ensure the current
        # iteration is also counted.
        ith = self.ith
        z_prev = self.zs[self.h]  # == from previous step
        f_h = self.fs[self.h + 1]  # current step

        z_h = ((ith - 1) / (ith)) * z_prev + (1 / (ith)) * f_h

        return z_h

    def _calculate_distance(self) -> float:
        """Calculate and store the distance to the pareto set for the current
        iteration

        """
        ds = 100 * (
            np.linalg.norm(self.zs[self.h + 1] - self.problem.nadir)
            / (np.linalg.norm(self.fs[self.h + 1] - self.problem.nadir))
        )
        return ds

    def initialize(
        self, itn: int = 5
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """Initialize the method by setting the initialization parameters and
        caluclating the initial bounds of the problem, the nadir and ideal
        point, if not defined in the problem. See initialization_requirements.

        Arguments:
            itn (int): Number of total iterations. Defaults to 5.

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]], float]: A tuple
            containing:
                np.ndarray: The current iteration point.
                List[Tuple[float, float]]: A list with tuples with the lower
                and upper bounds for the next iteration
                float: The distance of the current iteration point to the
                pareto optimal set.

        Raises:
            InteractiveMethodError: Wrong type of parameter or missing
            parameter in initialization_requirements.

        """
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

        self.h = 0
        # Skip the checks in the setter of ith
        self.__ith = self.itn

        self.zs = [None] * (self.itn + 1)
        self.zs[0] = self.problem.nadir

        self.lower_bounds = [None] * (self.itn + 1)
        self.lower_bounds[self.h] = self.problem.ideal

        self.upper_bounds = [None] * (self.itn + 1)
        self.upper_bounds[self.h] = self.problem.nadir

        self.xs = [None] * (self.itn + 1)

        self.fs = [None] * (self.itn + 1)

        self.ds = [0.0] * (self.itn + 1)

        self.asf.nadir_point = self.problem.nadir
        self.asf.utopian_point = self.problem.ideal - self.epsilon

        return (
            self.zs[self.h],
            list(zip(self.lower_bounds[self.h], self.upper_bounds[self.h])),
            self.ds[self.h],
        )

    def iterate(self) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
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
        # Calculate the preferential factors or use existing ones
        if not self.__short_step:
            if self.preference_percentages is not None:
                # use percentages to calcualte the new iteration point
                delta_q = self.preference_percentages / 100
                self.mu = 1 / (
                    delta_q
                    * (
                        self.problem.nadir
                        - (self.problem.ideal - self.epsilon)
                    )
                )

            elif self.preference_index_set is not None:
                # Use the relative importance to calcualte the new points
                print(self.preference_index_set)
                for (i, r) in enumerate(self.preference_index_set):
                    self.mu[i] = 1 / (
                        r
                        * (
                            self.problem.nadir[i]
                            - (self.problem.ideal[i] - self.epsilon)
                        )
                    )

            elif self.__use_previous_preference:
                # previous
                pass

            else:
                msg = "Could not compute the preferential factors."
                logger.debug(msg)
                raise InteractiveMethodError(msg)

        if not self.__short_step and not self.__use_previous_preference:
            # Take a normal step and calculate a new reference point
            # set the current iteration point as the reference point
            self.q = self.zs[self.h]

            # set the preferential factors in the underlaying asf
            self.asf.preferential_factors = self.mu

            # solve the ASF
            (solution, (objective, _)) = self.scalar_solver.solve(self.q)

            # Store the solution and corresponding objective vector
            self.xs[self.h + 1] = solution
            self.fs[self.h + 1] = objective[0]

            # calculate a new iteration point
            self.zs[self.h + 1] = self._calculate_iteration_point()

        elif not self.__step_back and self.__use_previous_preference:
            # Use the solution and objective of the last step
            self.xs[self.h + 1] = self.xs[self.h]
            self.fs[self.h + 1] = self.fs[self.h]
            self.zs[self.h + 1] = self._calculate_iteration_point()

        else:
            # Take a short step
            # Update the current iteration point
            self.zs[self.h + 1] = (
                0.5 * self.zs[self.h + 1] + 0.5 * self.zs[self.h]
            )

        # calculate the lower bounds for the next iteration
        if self.ith > 1:
            # last iteration, no need to calculate these
            self.lower_bounds[self.h + 1] = np.zeros(
                self.problem.n_of_objectives
            )

            self.__epsilon_solver.epsilons = self.zs[self.h + 1]

            for r in range(self.problem.n_of_objectives):
                (_, (objective, _)) = self.__epsilon_solver.solve(r)
                self.lower_bounds[self.h + 1][r] = objective[0][r]

                # set the upper bounds
                self.upper_bounds[self.h + 1] = self.zs[self.h]

        else:
            self.lower_bounds[self.h + 1] = [
                None
            ] * self.problem.n_of_objectives
            self.upper_bounds[self.h + 1] = [
                None
            ] * self.problem.n_of_objectives

        # Calculate the distance to the pareto optimal set
        self.ds[self.h + 1] = self._calculate_distance()

        return (
            self.zs[self.h + 1],
            list(
                zip(
                    self.lower_bounds[self.h + 1],
                    self.upper_bounds[self.h + 1],
                )
            ),
            self.ds[self.h + 1],
        )

    def interact(
        self,
        index_set: np.ndarray = None,
        percentages: np.ndarray = None,
        use_previous_preference: bool = False,
        new_remaining_iterations: Optional[int] = None,
        step_back: bool = False,
        short_step: bool = False,
    ) -> Union[int, Tuple[np.ndarray, np.ndarray]]:
        """Change the total number of iterations if supplied and take a step
        backwards if the DM wishes to do so.

        """
        if index_set is not None:
            self.preference_index_set = index_set
            self.__use_previous_preference = False

        elif percentages is not None:
            self.preference_percentages = percentages
            self.__use_previous_preference = False

        elif self.mu is not None and use_previous_preference:
            self.__use_previous_preference = use_previous_preference

        else:
            msg = (
                "Cannot use preferential infromation from previous "
                "iteration."
            )
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        if new_remaining_iterations is not None:
            self.ith = new_remaining_iterations

        if not step_back:
            self.__step_back = False
            # Advance the current iteration, if not the first iteration
            if not self.__first_iteration:
                self.ith -= 1
                self.h += 1
            else:
                self.__first_iteration = False

            if self.ith == 0:
                # Last iteration, terminate the solution
                return (self.xs[self.h], self.fs[self.h])

        else:
            if self.__first_iteration:
                msg = "Cannot take a backwards step on the first iteration."
                logger.debug(msg)
                raise InteractiveMethodError(msg)
            self.__step_back = True

        if short_step:
            if not step_back:
                msg = (
                    "Can take a short step only when stepping from the "
                    "previous point."
                )
                logger.debug(msg)
                raise InteractiveMethodError(msg)
        self.__short_step = short_step

        return self.ith
