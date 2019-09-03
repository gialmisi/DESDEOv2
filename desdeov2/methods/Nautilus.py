"""Various interactive multi-objective optimization methods belonging to the
NAUTILUS-family are defined here.

"""

import logging
import logging.config
from os import path
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans

from desdeov2.methods.InteractiveMethod import (
    InteractiveMethodBase,
    InteractiveMethodError,
)
from desdeov2.problem.Problem import (
    ProblemBase,
    ScalarDataProblem,
    ScalarMOProblem,
)
from desdeov2.solver.ASF import ReferencePointASF
from desdeov2.solver.NumericalMethods import NumericalMethodBase, ScipyDE
from desdeov2.solver.PointSolver import IdealAndNadirPointSolver
from desdeov2.solver.ScalarSolver import (
    ASFScalarSolver,
    EpsilonConstraintScalarSolver,
)
from desdeov2.utils.frozen import frozen

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


@frozen(logger)
class Nautilus(InteractiveMethodBase):
    """Implements the basic NAUTILUS methods as presented in `Miettinen 2010`_

    Attributes:
        epsilon (float): A small number used in calculating the utopian point.
        itn (int): Total number of iterations.
        h (int): Current iteration.
        ith (int): Number of remaining iterations.
        lower_bound (List[np.ndarray]): The lower bounds of reachable values
        from different iteration point.
        upper_bound (List[np.ndarray]): The upper bounds of reachable values
        from different iteration point.
        zs (List[np.ndarray]): Iteration points.
        q (np.ndarray): Current iteration point.
        xs (List[np.ndarray]): The solutions for each iteration.
        fs (List[np.ndarray]): The objective vector values for each iteration.
        ds (List[float]): The distance to the pareto front in each iteration.
        preference_index_set (np.ndarray): The current DM's preference as index
        sets
        preference_percentages (np.ndarray): The current DM's preferences as
        percentages.
        mu (np.ndarray): The current preferential factors calculated from the
        DM preferences.
        scalar_solver (ASFScalarSolver): A solver for solving scalarized
        problems using ASFs.
        asf (ReferencePointASF): Reference point based ASF used in the
        scalar_solver

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

        self.__numerical_method: NumericalMethodBase = ScipyDE()
        self.__scalar_solver: ASFScalarSolver = ASFScalarSolver(
            self.problem, self.__numerical_method
        )  # noqa
        self.__asf: ReferencePointASF = ReferencePointASF(None, None, None)
        self.__scalar_solver.asf = self.__asf

        self.__epsilon_solver: EpsilonConstraintScalarSolver = EpsilonConstraintScalarSolver(  # noqa: E501
            self.problem, self.__numerical_method
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

        Args:
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
        """Set the number of remaining iterations. Should be less than the current
        remaining iterations

        Args:
            val (int): New number of iterations to carry out.

        Raises:
            InteractiveMethodError: val is either negative or greater than the
            current number of remaining iterations.

        """
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
        return self.__preference_index_set

    @preference_index_set.setter
    def preference_index_set(self, val: np.ndarray):
        """Set the indexes to rank each of the objectives in order of
        importance.

        Args:
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
        self.__preference_index_set = val

    @property
    def preference_percentages(self) -> np.ndarray:
        return self.__preference_percentages

    @preference_percentages.setter
    def preference_percentages(self, val: np.ndarray):
        """Set the percentages to descripe how to improve each objective.

        Args:
            val (np.ndarray): A 1D-vector containing percentages corresponding
            to each objective.

        Raises:
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
        """Calculate the new iteration point based on the current iteration's
        iteration point and pareto optimal solution.

        Returns:
            np.ndarray: New iteration point

        """
        # Number of iterations left. The plus one is to ensure the current
        # iteration is also counted.
        ith = self.ith
        z_prev = self.zs[self.h]  # == from previous step
        f_h = self.fs[self.h + 1]  # current step

        z_h = ((ith - 1) / (ith)) * z_prev + (1 / (ith)) * f_h

        return z_h

    def _calculate_distance(self) -> float:
        """Calculate the distance to the pareto optimal front in the current
        iteration.

        Returns:
            float: The distance to the pareto front ranging from 0-100, 100
            being the closest distance to the front.

        """
        ds = 100 * (
            np.linalg.norm(self.zs[self.h + 1] - self.problem.nadir)
            / (np.linalg.norm(self.fs[self.h + 1] - self.problem.nadir))
        )
        return ds

    def initialize(  # type: ignore
        self, itn: int = 5
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """Initialize the method by setting the initialization parameters and
        caluclating the initial bounds of the problem, the nadir and ideal
        point, if not defined in the problem.

        Args:
            itn (int): Number of total iterations. Defaults to 5.

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]], float]: A tuple
            containing:
                np.ndarray: The current iteration point.
                List[Tuple[float, float]]: A list with tuples with the lower
                and upper bounds for the next iteration
                float: The distance of the current iteration point to the
                pareto optimal set.

        """
        self.itn = itn

        # Check if the ideal and nadir points are set
        if self.problem.ideal is None and self.problem.nadir is None:
            # Both missing, compute both
            solver = IdealAndNadirPointSolver(
                self.problem, self.__numerical_method
            )
            self.problem.ideal, self.problem.nadir = solver.solve()

        elif self.problem.ideal is None:
            # ideal missing, compute it
            solver = IdealAndNadirPointSolver(
                self.problem, self.__numerical_method
            )
            self.problem.ideal, _ = solver.solve()

        elif self.problem.nadir is None:
            # nadir missing, compute it
            solver = IdealAndNadirPointSolver(
                self.problem, self.__numerical_method
            )
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

        self.asf.nadir = self.problem.nadir
        self.asf.utopian_point = self.problem.ideal - self.epsilon

        return (
            self.zs[self.h],
            list(zip(self.lower_bounds[self.h], self.upper_bounds[self.h])),
            self.ds[self.h],
        )

    def iterate(self) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """Iterate once according to the user preference given in the
        interaction phase.

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]], float]: A tuple
            containing:
                np.ndarray: The current iteration point.
                List[Tuple[float, float]]: A list with tuples with the lower
                and upper bounds for the next iteration
                float: The distance of the current iteration point to the
                pareto optimal set.

        Raises:
            InteractiveMethodError: If the preferential factors can't be
            computed

        Note:
            The current iteration is to be interpreted as self.h + 1, since
            incrementation of the current iteration happens in the interaction
            phase.

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
        """Handle user preference and set appropiate flags for the next iteration.

        Args:
            index_set (np.ndarray): An array with integers describing the
            relative importance of each objective. The integers vary between 1
            and the maximum number of objectives in the problem.
            percentages (np.ndarray): Percentages describing the absolute
            importance of each objective. The sum of these must equal 100.
            use_previous_preference (bool): Use the preference
            infromation. Cannot be true during the first iteration.
            defined in the last iteration. Defaults to false.
            new_remaining_iterations (int): Set a new number of remaining
            iterations to be carried. Must be positive and not exceed the
            current number of iterations left.
            step_back (bool): Step from the previous point in the next
            iteration. Cannot step back from first iteration.
            short_step (bool): When step_back, take a shorter step in the same
            direction as in the previous iteration from the previous
            iteration's iteration point. Can only short step when stepping
            back.

        Returns:
            Union[int, Tuple[np.ndarray, np.ndarray]]: The number of remaining
            iteration. If this function is envoked after the last iteration,
            returns a tuple with the optimal solution and objective values.

        Raises:
            InteractiveMethodError: Some of the arguments are not set
            correctly. See the documentation for the arguments.

        """
        if index_set is not None:
            self.preference_index_set = index_set
            self.__use_previous_preference = False

        elif percentages is not None:
            self.preference_percentages = percentages
            self.__use_previous_preference = False

        elif self.mu is not None and use_previous_preference:
            self.__use_previous_preference = use_previous_preference

        elif step_back and short_step:
            # no preference needed for short step
            pass

        else:
            msg = "Cannot figure out preference infromation."
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


@frozen(logger)
class ENautilus(InteractiveMethodBase):
    """Implements the enhanced Nautilus variant, E-Nautilus originally
    presented in `Ruiz 2015`_

    Args:
        pareto_front (np.ndarray): The representation of a pareto front for a
        problem to be solved.
        objective_vectors (np.ndarray): The objective vectors corresponding to
        the pareto front.
        nadir (np.ndarray): The nadir point.
        ideal (np.ndarray): The ideal point.
        n_iters (int): The number of total iterations.
        n_points (int): The number of intermediate points to be generated.
        zshi (np.ndarray): The intermediate points at each iteration.
        h (int): The current iteration.
        ith (int): The iterations left.
        par_sub (List[np.ndarray]): The subspace of the pareto front reachable
        during each iteration.
        par_obj (List[np.ndarray]): The subspace of the reachable objective
        vector values during each iteration.
        fhilo (np.ndarray): The lower bounds of the reachable set from each
        intermediate point.
        d (np.ndarray): How close to the pareto front the each of the
        intermediate points are for each iteration.
        zpref (np.ndarray): The preferred point at each iteration.

    .. _Ruiz 2015:
        Ruiz A. B.; Sindhya K.; Miettinen K.; Ruiz F. & Luque M.
        E-NAUTILUS: A decision support system for complex multiobjective
        optimization problems based on the NAUTILUS method
        Europen Joural of Operational Research, 2015, 246, 218-231
    """

    def __init__(self, problem: ProblemBase):
        super().__init__(problem)

        if isinstance(problem, ScalarMOProblem):
            msg = (
                "Currently E-NAUTILUS works only with the "
                "ScalarDataProblem problem class or by sypplying "
                "the pareto set and objectives manually."
            )
            logger.error(msg)
            raise NotImplementedError(msg)

        # The full pareto front
        self.__pareto_front: np.ndarray = None
        # Corresponding pareto optimal objective vectors for the full front
        self.__objective_vectors: np.ndarray = None
        self.__nadir: np.ndarray = None
        self.__ideal: np.ndarray = None
        # number of total iterations
        self.__n_iters: int = 0
        # number of points to be shown to the DM after each iteration
        self.__n_points: int = 0
        # the i (n_points) intermediate points at step h
        self.__zshi: np.ndarray = None
        # current iteration
        self.__h: int = 0
        # iterations left
        self.__ith: int = 0
        # the subset of the reachable pareto optimal solutions
        # during each iteration
        self.__par_sub: List[np.ndarray] = []
        # the subset of the reachable pareto optimal objective vectors
        # during each iteration
        self.__obj_sub: List[np.ndarray] = []
        # lower bounds of the reachable set from each intermediate point
        self.__fhilo: np.ndarray = None
        # closeness to the pareto front
        self.__d: np.ndarray = None
        # prefered point
        self.__zpref: np.ndarray = None

    @property
    def pareto_front(self) -> np.ndarray:
        return self.__pareto_front

    @pareto_front.setter
    def pareto_front(self, val: np.ndarray):
        self.__pareto_front = val

    @property
    def objective_vectors(self) -> np.ndarray:
        return self.__objective_vectors

    @objective_vectors.setter
    def objective_vectors(self, val: np.ndarray):
        self.__objective_vectors = val

    @property
    def nadir(self) -> np.ndarray:
        return self.__nadir

    @nadir.setter
    def nadir(self, val: np.ndarray):
        """Set the nadir point.

        Args:
            val (np.ndarray): The nadir point.

        Raises:
            InteractiveMethodError: The nadir point is of the wrong dimensions.

        """
        if len(val) != self.objective_vectors.shape[1]:
            msg = (
                "The nadir point's length '{}' must match the number of "
                "objectives '{}' (columns) specified in the given "
                "pareto front."
            ).format(len(val), self.objective_vectors.shape[1])
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        self.__nadir = val

    @property
    def ideal(self) -> np.ndarray:
        return self.__ideal

    @ideal.setter
    def ideal(self, val: np.ndarray):
        """Set the ideal point.

        Args:
            val (np.ndarray): The ideal point.

        Raises:
            InteractiveMethodError: The ideal point is of the wrong dimensions.

        """
        if len(val) != self.objective_vectors.shape[1]:
            msg = (
                "The ideal point's length '{}' must match the number of "
                "objectives '{}' (columns) specified in the given "
                "pareto front."
            ).format(len(val), self.objective_vectors.shape[1])
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        self.__ideal = val

    @property
    def n_iters(self) -> int:
        return self.__n_iters

    @n_iters.setter
    def n_iters(self, val: int):
        """Set the total number of iterations to be carried out.

        Args:
            val (int): The number of iterations.

        Raises:
            InteractiveMethodError: The number of iterations is non-positive.

        """
        if val < 1:
            msg = "Number of iterations must be greater than zero."
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        self.__n_iters = val

    @property
    def n_points(self) -> int:
        return self.__n_points

    @n_points.setter
    def n_points(self, val: int):
        """The number of points to be presented to the DM during each iteration.

        Args:
            val (int): Number of points to be shown.

        Raises:
            InteractiveMethodError: The number of points is non-positive.

        """
        if val < 1:
            msg = "The number of points shown must be greater than zero."
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        self.__n_points = val

    @property
    def zshi(self) -> np.ndarray:
        return self.__zshi

    @zshi.setter
    def zshi(self, val: np.ndarray):
        self.__zshi = val

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
    def par_sub(self) -> List[np.ndarray]:
        return self.__par_sub

    @par_sub.setter
    def par_sub(self, val: List[np.ndarray]):
        self.__par_sub = val

    @property
    def obj_sub(self) -> List[np.ndarray]:
        return self.__obj_sub

    @obj_sub.setter
    def obj_sub(self, val: List[np.ndarray]):
        self.__obj_sub = val

    @property
    def fhilo(self) -> np.ndarray:
        return self.__fhilo

    @fhilo.setter
    def fhilo(self, val: np.ndarray):
        self.__fhilo = val

    @property
    def d(self) -> np.ndarray:
        return self.__d

    @d.setter
    def d(self, val: np.ndarray):
        self.__d = val

    @property
    def zpref(self) -> np.ndarray:
        return self.__zpref

    @zpref.setter
    def zpref(self, val: np.ndarray):
        self.__zpref = val

    def initialize(  # type: ignore
        self,
        n_iters: int,
        n_points: int,
        pareto_front: Optional[np.ndarray] = None,
        objective_vectors: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize the method with the input data required by E-NAUTILUS by
        either the given problem in the initializer of the class or explicitly
        given pareto and objective data.

        Args:
            n_iters (int): The number of total iterations to be carried out.
            n_points (int): The number of points to be shown to the DM each
            iteration.
            pareto_front (Optional[np.ndarray]): Vectors preresenting the
            pareto optimal solutions of a MOO problem.
            objective_vectors (Optional[np.ndarray]): The objective vectors
            that correspond to the pareto optimal solutions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                np.ndarray: The nadir point of the problem.
                np.ndarray: The ideal point of the problem.
        Raises:
            NotImplementedError: The problem type must be ScalarDataProblem, if
            not, raise this.

        Note:
            Only support solving ScalarDataProblem at the moment.

        """
        if isinstance(self.problem, ScalarDataProblem):
            self.pareto_front = self.problem.decision_vectors
            self.objective_vectors = self.problem.objective_vectors
        else:
            msg = (
                "ENautilus currently supports only "
                "solving ScalarDataProblems"
            )
            logger.error(msg)
            raise NotImplementedError(msg)

        self.n_iters = n_iters
        self.n_points = n_points

        # find the nadir and ideal points
        self.nadir = np.max(self.objective_vectors, axis=0)
        self.ideal = np.min(self.objective_vectors, axis=0)

        # initialize the intermediate points, bounds, and distances
        self.zshi = np.full(
            (self.n_iters, self.n_points, self.objective_vectors.shape[1]),
            np.nan,
            dtype=np.float,
        )

        self.fhilo = np.full(
            (self.n_iters, self.n_points, self.objective_vectors.shape[1]),
            np.nan,
            dtype=np.float,
        )

        self.d = np.full((self.n_iters, self.n_points), np.nan, dtype=np.float)

        # initialize the reachable subsets (None at the zeroth position,
        # because we want to use h to access them)
        self.par_sub = [None] * (self.n_iters)
        self.obj_sub = [None] * (self.n_iters)

        self.h = 0
        self.ith = self.n_iters
        self.par_sub[self.h] = self.pareto_front
        self.obj_sub[self.h] = self.objective_vectors
        self.zpref = self.nadir

        return self.nadir, self.ideal

    def iterate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the intermediate points and the lower bounds of
        the reachable solutions from each point in the next iteration
        according to a preference point specified by the decision maker during
        the previous interaction. During the first iteration, the nadir point
        is used instead.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                np.ndarray: The intermediate points.
                np.ndarray: The lower bounds of the reachable values from each
                intermediate point.

        Note:
            If the requested number of points cannot be generated, the
            resulting arrays containing the intermediate points and bounds will
            be padded with NaNs.

        """
        if self.ith <= 1:
            logger.info(
                (
                    "Last iteration, or trying to iterate past the last "
                    "iteration. Please call "
                    "interact with the final preference to generate the final "
                    "solution. Returning the most recent "
                    "intermediate points."
                )
            )
            return self.zshi[self.h - 1], self.fhilo[self.h - 1]

        # Use clustering to find the most representative points
        if self.n_points <= len(self.obj_sub[self.h]):
            kmeans = KMeans(n_clusters=self.n_points)
            kmeans.fit(self.obj_sub[self.h])
            zbars = kmeans.cluster_centers_
        else:
            # the subspace has less or an equal amount of points to the number
            # points to be shown, just use the subspace
            msg = (
                "Could not generate the requested amount of intermediate "
                "points '{}'. Generating only '{}' points"
            ).format(self.n_points, len(self.obj_sub[self.h]))
            logger.info(msg)
            zbars = self.obj_sub[self.h]

        # calculate the intermediate points
        self.zshi[self.h][0 : len(zbars)] = (  # noqa
            (self.ith - 1) / self.ith
        ) * self.zpref + (1 / self.ith) * zbars

        # calculate the lower bounds
        for r in range(self.objective_vectors.shape[1]):
            col_mask = np.full(self.objective_vectors.shape[1], True)
            col_mask[r] = False
            for i in range(len(zbars)):
                mask = np.all(
                    self.zshi[self.h, i, col_mask]
                    >= self.obj_sub[self.h][:, col_mask],
                    axis=1,
                )

                # if the mask if full of false, do nothing
                if not np.all(~mask):
                    self.fhilo[self.h, i, r] = np.min(
                        self.obj_sub[self.h][mask, r]
                    )

        # calculate the distances to the pareto front for each representative
        # point
        self.d[self.h][0 : len(zbars)] = (  # noqa
            np.linalg.norm(
                self.zshi[self.h][0 : len(zbars)] - self.nadir, axis=1  # noqa
            )
            / np.linalg.norm(zbars - self.nadir, axis=1)
        ) * 100

        return self.zshi[self.h], self.fhilo[self.h]

    def interact(  # type: ignore
        self, preferred_point: np.ndarray, lower_bounds: np.ndarray
    ) -> Union[int, Tuple[np.ndarray, np.ndarray]]:
        """Specify the next preferred point from which to iterate in the next
        iteration. The lower bounds of the reachable values from the preferred
        point are also expected. This point does not necessarely need to be a
        point returned by the iterate method in this class.

        Args:
            preferred_point (np.ndarray): An objective value vector
            representing the preferred point.
            lower_bounds (np.ndarray): The lower bounds of the reachable values
            from the preferred point.

        Returns:
            Union[int, Tuple[np.ndarray, np.ndarray]]: The number of iterations
                left, if not invoked on the last iteration. Otherwise a tuple
                containing: np.ndarray: The final pareto optimal solution.
                np.ndarray: The corresponding objevtive vector to the pareto
                optimal solution.

        Raises:
            InteractiveMethodError: The dimensions of either the preferred
            point or the lower bounds of the reachable values are incorrect.

        """
        if len(preferred_point) != self.objective_vectors.shape[1]:
            # check that the dimensions of the given points are correct
            msg = (
                "The dimensions of the prefered point '{}' do not match "
                "the shape of the objective vectors '{}'."
            ).format(len(preferred_point), self.objective_vectors.shape[1])
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        if len(lower_bounds) != self.objective_vectors.shape[1]:
            msg = (
                "The dimensions of the lower bounds for the prefered "
                "point '{}' do not match "
                "the shape of the objective vectors '{}'."
            ).format(len(lower_bounds), self.objective_vectors.shape[1])
            logger.debug(msg)
            raise InteractiveMethodError(msg)

        self.zpref = preferred_point

        if self.ith <= 1:
            # stop the algorithm and return the final solution and the
            # corresponding objective vector
            idx = np.linalg.norm(
                self.obj_sub[self.h] - self.zpref, axis=1
            ).argmin()
            self.ith = 0
            return self.par_sub[self.h][idx], self.obj_sub[self.h][idx]

        # Calculate the new reachable pareto solutions and objective vectors
        # from zpref
        cond1 = np.all(
            np.less_equal(lower_bounds, self.obj_sub[self.h]), axis=1
        )
        cond2 = np.all(np.less_equal(self.obj_sub[self.h], self.zpref), axis=1)

        indices = (cond1 & cond2).nonzero()

        self.obj_sub[self.h + 1] = self.obj_sub[self.h][indices]
        self.par_sub[self.h + 1] = self.par_sub[self.h][indices]

        self.ith -= 1
        self.h += 1

        return self.ith
