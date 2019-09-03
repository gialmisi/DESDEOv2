"""Various interactive multiobjective optimization methods belonging to them
NIMBUS-family are defined here

"""

import logging
import logging.config
from copy import deepcopy
from os import path
from typing import List, Optional, Tuple, Union

import numpy as np

from desdeov2.methods.InteractiveMethod import (
    InteractiveMethodBase,
    InteractiveMethodError,
)
from desdeov2.problem.Constraint import ScalarConstraint
from desdeov2.problem.Problem import (
    ProblemBase,
    ScalarDataProblem,
    ScalarMOProblem,
)
from desdeov2.solver.ASF import (
    AugmentedGuessASF,
    MaxOfTwoASF,
    PointMethodASF,
    StomASF,
)
from desdeov2.solver.NumericalMethods import DiscreteMinimizer
from desdeov2.solver.ScalarSolver import ASFScalarSolver
from desdeov2.utils.frozen import frozen

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


@frozen(logger)
class SNimbus(InteractiveMethodBase):
    """Implements the synchronous NIMBUS variant as defined in
       `Miettinen 2016`_

    Attributes:
        pareto_front (np.ndarray): The representation of the pareto front of
        the problem to be solved.
        objective_vectors (np.ndarray): The objective vector values
        corresponding to the representation of the pareto front.
        classifications (List[Tuple[str, Optional[float]]]): Tuples
        representing the classifications of each objective funtion. The second
        element of the tuple is an extra information needed by some
        classifications.
        nadir (np.ndarray): The nadir point.
        ideal (np.ndarray): The ideal point.
        cind (int): current index of the objective solution (used to map it
        back to the decision variables)
        current_point (np.ndarray): The currently selected objective vector.
        archive (List[np.ndarray]): Archived solutions.
        n_points (int): Number of points to be generated each iteration.
        first_iteration (bool): Indicated whether the method has not been
        iterated yet.
        generate_intermediate (bool): Should intermediate points be generated
        in the next iteration?
        search_between_points (Tuple[np.ndarray]): The two points between the
        intermediate points should be generated.
        n_intermediate_solutions (int): Number of intermediate solutions to be
        generated between two objective vectors.

    .. _Miettinen 2006:
        Mietttinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922

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
        # The objective vector values correspond to the front
        self.__objective_vectors: np.ndarray = None
        # classifications of the objective functions
        self.__classifications: List[Tuple[str, Optional[float]]] = []
        self.__available_classifications: List[str] = [
            "<",
            "<=",
            "=",
            ">=",
            "0",
        ]
        # index sets to keep track how each objective should change
        self.__ind_set_lt: List[int] = []
        self.__ind_set_lte: List[int] = []
        self.__ind_set_eq: List[int] = []
        self.__ind_set_gte: List[int] = []
        self.__ind_set_free: List[int] = []
        # bounds
        self.__aspiration_levels: np.ndarray = None
        self.__upper_bounds: np.ndarray = None
        # nadir
        self.__nadir: np.ndarray = None
        # ideal
        self.__ideal: np.ndarray = None
        # current index of the solution used
        self.__cind: np.int = 0
        # currently selected objective vector
        self.__current_point: np.ndarray = None
        # solution archive
        self.__archive: List[Tuple[np.ndarray, np.ndarray]] = []
        # number of point to be generated
        self.__n_points: int = 0
        # flag to represent if first iteration
        self.__first_iteration: bool = True
        # flag to generate intermediate points
        self.__generate_intermediate: bool = False
        # points between intermediate solutions are explored
        self.__search_between_points: Tuple[np.ndarray, np.ndarray] = (
            None,
            None,
        )  # noqa
        # number of intermediate points to be generated
        self.__n_intermediate_solutions: int = 0
        # subproblems
        self.__sprob_1: Optional[ScalarDataProblem] = None
        self.__sprob_2: Optional[ScalarDataProblem] = None
        self.__sprob_3: Optional[ScalarDataProblem] = None
        self.__sprob_4: Optional[ScalarDataProblem] = None

        # solvers for each subproblem
        self.__solver_1: Optional[ASFScalarSolver] = None
        self.__solver_2: Optional[ASFScalarSolver] = None
        self.__solver_3: Optional[ASFScalarSolver] = None
        self.__solver_4: Optional[ASFScalarSolver] = None

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
    def classifications(self) -> List[Tuple[str, Optional[float]]]:
        return self.__classifications

    @classifications.setter
    def classifications(self, val: List[Tuple[str, Optional[float]]]):
        """Parses classifications and checks if they are sensical. See
        `Miettinen 2016`_

        Args:
            val (List[Tuple, Optional[float]]): The classificaitons. The first
            element is the class and the second element is auxilliary
            information needed by some classifications.

        Raises:
            InteractiveMethodError: The classifications given are ill-formed.

        """
        if len(val) != self.objective_vectors.shape[1]:
            msg = (
                "Each of the objective functions must be classified. Check "
                "that '{}' has a correct amount (in this case {}) of "
                "elements in it."
            ).format(val, self.objective_vectors.shape[1])
            logger.error(msg)
            raise InteractiveMethodError(msg)

        if not all(
            [cls[0] in self.__available_classifications for cls in val]
        ):
            msg = (
                "Check the given classifications '{}'. The first element of "
                "each tuple should be found in '{}'"
            ).format(val, self.__available_classifications)
            logger.error(msg)
            raise InteractiveMethodError(msg)

        clss = [cls[0] for cls in val]
        if not (
            ("<" in clss or "<=" in clss) and (">=" in clss or "0" in clss)
        ):
            msg = (
                "Check the calssifications '{}'. At least one of the "
                "objectives should able to be improved and one of the "
                "objectives should be able to deteriorate."
            ).format(val)
            logger.error(msg)
            raise InteractiveMethodError(msg)

        for (ind, cls) in enumerate(val):
            if cls[0] == "<=":
                if not cls[1] < self.current_point[ind]:
                    msg = (
                        "For the '{}th' objective, the aspiration level '{}' "
                        "must be smaller than the current value of the "
                        "objective '{}'."
                    ).format(ind, cls[1], self.current_point[ind])
                    logger.error(msg)
                    raise InteractiveMethodError(msg)
            elif cls[0] == ">=":
                if not cls[1] > self.current_point[ind]:
                    msg = (
                        "For the '{}th' objective, the upper bound '{}' "
                        "must be greater than the current value of the "
                        "objective '{}'."
                    ).format(ind, cls[1], self.current_point[ind])
                    logger.error(msg)
                    raise InteractiveMethodError(msg)

        self.__classifications = val

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
    def cind(self) -> int:
        return self.__cind

    @cind.setter
    def cind(self, val: int):
        self.__cind = val

    @property
    def archive(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return self.__archive

    @archive.setter
    def archive(self, val: List[Tuple[np.ndarray, np.ndarray]]):
        self.__archive = val

    @property
    def current_point(self) -> np.ndarray:
        return self.__current_point

    @current_point.setter
    def current_point(self, val: np.ndarray):
        """Set the current point for the algorithm. The dimensions of the
        current point must match the dimensions of the row vectors in
        objective_vectors.

        Args:
            val(np.ndarray): The current point.

        Raises:
            InteractiveMethodError: The current point's dimension does not
            match that of the given pareto optimal objective vectors.

        """
        if len(val) != self.objective_vectors.shape[1]:
            msg = (
                "Current point dimensions '{}' don't match the objective"
                " vector dimensions '{}'"
            ).format(len(val), self.objective_vectors.shape[1])
            logger.error(msg)
            raise InteractiveMethodError(msg)

        self.__current_point = val

    @property
    def n_points(self) -> int:
        return self.__n_points

    @n_points.setter
    def n_points(self, val: int):
        """Set the desired number of solutions to be generated each
        iteration. Must be between 1 and 4 (inclusive)

        Args:
            val(int): The number of points to be generated.

        Raises:
            InteractiveMethodError: The number of points to be generated is not
            between 1 and 4.

        """
        if val < 1 or val > 4:
            msg = (
                "The given number '{}' of solutions to be generated is not"
                " between 1 and 4 (inclusive)"
            ).format(val)
            logger.error(msg)
            raise InteractiveMethodError(msg)
        self.__n_points = val

    @property
    def first_iteration(self) -> bool:
        return self.__first_iteration

    @first_iteration.setter
    def first_iteration(self, val: bool):
        self.__first_iteration = val

    @property
    def generate_intermediate(self) -> bool:
        return self.__generate_intermediate

    @generate_intermediate.setter
    def generate_intermediate(self, val: bool):
        self.__generate_intermediate = val

    @property
    def search_between_points(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.__search_between_points

    @search_between_points.setter
    def search_between_points(self, val: Tuple[np.ndarray, np.ndarray]):
        if len(val) != 2:
            msg = (
                "To generate intermediate points, two points must be "
                "specified. Number of given points were {}"
            ).format(len(val))
            logger.error(msg)
            raise InteractiveMethodError(msg)
        self.__search_between_points = val

    @property
    def n_intermediate_solutions(self) -> int:
        return self.__n_intermediate_solutions

    @n_intermediate_solutions.setter
    def n_intermediate_solutions(self, val: int):
        if val < 1:
            msg = (
                "Number of intermediate points to be generated must be "
                "positive definitive. Given number of points {}."
            ).format(val)
            logger.error(msg)
            raise InteractiveMethodError(msg)

        self.__n_intermediate_solutions = val

    def _sort_classsifications(self):
        """Sort the objective indices into their corresponding sets and save
        the aspiration and upper bounds set in the classifications.

        Raises:
            InteractiveMethodError: A classification is found to be ill-formed.

        """
        # empty the sets and bounds
        self.__ind_set_lt = []
        self.__ind_set_lte = []
        self.__ind_set_eq = []
        self.__ind_set_gte = []
        self.__ind_set_free = []
        aspiration_levels = []
        upper_bounds = []
        for (ind, cls) in enumerate(self.classifications):
            if cls[0] == "<":
                self.__ind_set_lt.append(ind)
            elif cls[0] == "<=":
                self.__ind_set_lte.append(ind)
                aspiration_levels.append(cls[1])
            elif cls[0] == "=":
                self.__ind_set_eq.append(ind)
            elif cls[0] == ">=":
                self.__ind_set_gte.append(ind)
                upper_bounds.append(cls[1])
            elif cls[0] == "0":
                self.__ind_set_free.append(ind)
            else:
                msg = (
                    "Check that the classification '{}' is correct."
                ).format(cls)
                logger.error(msg)
                raise InteractiveMethodError(msg)
        self.__aspiration_levels = np.array(aspiration_levels)
        self.__upper_bounds = np.array(upper_bounds)

    def _create_reference_point(self) -> np.ndarray:
        """Create a reference point indicating the DM's preferences using the
        classifications given by the DM.

        Returns:
            np.ndarray: The reference point.

        """
        ref_point = np.zeros(self.objective_vectors.shape[1])
        ref_point[self.__ind_set_lt] = self.ideal[self.__ind_set_lt]
        ref_point[self.__ind_set_lte] = self.__aspiration_levels
        ref_point[self.__ind_set_eq] = self.current_point[self.__ind_set_eq]
        ref_point[self.__ind_set_gte] = self.__upper_bounds
        ref_point[self.__ind_set_free] = self.nadir[self.__ind_set_free]

        return ref_point

    def _create_intermediate_reference_points(self) -> np.ndarray:
        f1 = self.search_between_points[0]
        f2 = self.search_between_points[1]
        points = []

        # The unnormalized vector pointing from f1 to f2
        normal = f2 - f1

        # fraction of the normal
        step_vec = normal / (self.n_intermediate_solutions + 1)
        points.append(f1 + step_vec)

        for i in range(1, self.n_intermediate_solutions):
            points.append(points[i - 1] + step_vec)

        return np.array(points)

    def initialize(  # type: ignore
        self, n_solutions: int, starting_point: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Initialize the method and return the starting objective vector.

        Args:
            n_solutions(int): The number of solutions to be generated each
            iteration.
            starting_point(Optional[np.ndarray]): An objective vector to
            function as the starting point for synchronous NIMBUS.
            pareto_front(Optional[np.ndarray]): A 2D array representing the
            pareto optimal solutions of the problem to be solved.
            objective_vectors(Optional[np.ndarray]): The objective vectors
            corresponding to each pareto optimal solution.

        Returns:
            np.ndarray: The starting point of the algorithm.

        Raises:
            NotImplementedError: The problem is something else than
            ScalarDataProblem

        Note:
            The data to be used must be available in the underlying problem OR
            given explicitly.

        """
        self.n_points = n_solutions

        if isinstance(self.problem, ScalarDataProblem):
            self.pareto_front = self.problem.decision_vectors
            self.objective_vectors = self.problem.objective_vectors

        else:
            msg = "Only supoorts solving for SacalarDataProblem at the moment."
            logger.error(msg)
            raise NotImplementedError(msg)

        self.nadir = np.max(self.objective_vectors, axis=0)
        self.ideal = np.min(self.objective_vectors, axis=0)

        self.__classifications = [("", None)] * self.objective_vectors.shape[1]

        # if the starting point has been specified, use that. Otherwise, use
        # random one.
        if starting_point is not None:
            self.current_point = starting_point

        else:
            rind = np.random.randint(0, len(self.objective_vectors))
            self.current_point = self.objective_vectors[rind]

        return self.current_point

    def iterate(
        self
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate according to the preferences given by the DM in the
        interaction phase.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            Returns the current point for the first iteration. For the
            following iterations, returns the decision vectors, the objective
            vectors values and the current archive of saved points.

        """
        # if first iteration, just return the starting point
        if self.first_iteration:
            self.first_iteration = False
            return self.current_point

        res_all_xs: List[np.ndarray] = []
        res_all_fs: List[np.ndarray] = []

        if self.generate_intermediate:
            # generate n points between two previous points
            # can reuse the solver for subrpoblem 3 here
            if self.__sprob_3 is None:
                # create the subproblem and solver
                self.__sprob_3 = ScalarDataProblem(
                    self.pareto_front, self.objective_vectors
                )
                self.__sprob_3.nadir = self.nadir
                self.__sprob_3.ideal = self.ideal

                self.__solver_3 = ASFScalarSolver(
                    self.__sprob_3, DiscreteMinimizer()
                )
                self.__solver_3.asf = PointMethodASF(self.nadir, self.ideal)

            z_bars = self._create_intermediate_reference_points()
            for z in z_bars:
                res = self.__solver_3.solve(z)  # type: ignore
                res_all_xs.append(res[0])
                res_all_fs.append(res[1][0])

            # always require and explicit request from the DM to generate
            # intermediate points
            self.generate_intermediate = False

        else:
            # solve the subproblems normally
            if self.n_points >= 1:
                # solve the desired number of ASFs
                self._sort_classsifications()
                z_bar = self._create_reference_point()

                # subproblem 1
                if self.__sprob_1 is None:
                    # create the subproblem and solver
                    self.__sprob_1 = ScalarDataProblem(
                        self.pareto_front, self.objective_vectors
                    )
                    self.__sprob_1.nadir = self.nadir
                    self.__sprob_1.ideal = self.ideal

                    self.__solver_1 = ASFScalarSolver(
                        self.__sprob_1, DiscreteMinimizer()
                    )
                    self.__solver_1.asf = MaxOfTwoASF(
                        self.nadir, self.ideal, [], []
                    )

                # set the constraints for the 1st subproblem
                sp1_all_cons = []
                sp1_cons1_idx = np.sort(
                    (
                        self.__ind_set_lt
                        + self.__ind_set_lte
                        + self.__ind_set_eq
                    )
                )

                if len(sp1_cons1_idx) > 0:
                    sp1_cons1_f = lambda _, fs: np.where(  # noqa
                        np.all(
                            fs[:, sp1_cons1_idx]
                            <= self.current_point[sp1_cons1_idx],
                            axis=1,
                        ),
                        np.ones(len(fs)),
                        -np.ones(len(fs)),
                    )
                    sp1_cons1 = ScalarConstraint(
                        "sp1_cons1",
                        self.pareto_front.shape[1],
                        self.objective_vectors.shape[1],
                        sp1_cons1_f,
                    )
                    sp1_all_cons.append(sp1_cons1)

                sp1_cons2_idx = self.__ind_set_gte
                if len(sp1_cons2_idx) > 0:
                    sp1_cons2_f = lambda _, fs: np.where(  # noqa
                        np.all(
                            fs[:, sp1_cons2_idx] <= self.__upper_bounds, axis=1
                        ),
                        np.ones(len(fs)),
                        -np.ones(len(fs)),
                    )
                    sp1_cons2 = ScalarConstraint(
                        "sp1_cons2",
                        self.pareto_front.shape[1],
                        self.objective_vectors.shape[1],
                        sp1_cons2_f,
                    )
                    sp1_all_cons.append(sp1_cons2)

                self.__sprob_1.constraints = sp1_all_cons

                # solve the subproblem
                self.__solver_1.asf.lt_inds = self.__ind_set_lt  # type: ignore
                self.__solver_1.asf.lte_inds = (  # type: ignore
                    self.__ind_set_lte
                )  # type: ignore, # noqa

                sp1_reference = np.zeros(self.objective_vectors.shape[1])
                sp1_reference[self.__ind_set_lte] = self.__aspiration_levels

                res1 = self.__solver_1.solve(sp1_reference)  # type: ignore
                res_all_xs.append(res1[0])
                res_all_fs.append(res1[1][0])

            if self.n_points >= 2:
                # subproblem 2
                if self.__sprob_2 is None:
                    # create the subproblem and solver
                    self.__sprob_2 = ScalarDataProblem(
                        self.pareto_front, self.objective_vectors
                    )
                    self.__sprob_2.nadir = self.nadir
                    self.__sprob_2.ideal = self.ideal

                    self.__solver_2 = ASFScalarSolver(
                        self.__sprob_2, DiscreteMinimizer()
                    )
                    self.__solver_2.asf = StomASF(self.ideal)

                res2 = self.__solver_2.solve(z_bar)  # type: ignore
                res_all_xs.append(res2[0])
                res_all_fs.append(res2[1][0])

            if self.n_points >= 3:
                # subproblem 3
                if self.__sprob_3 is None:
                    # create the subproblem and solver
                    self.__sprob_3 = ScalarDataProblem(
                        self.pareto_front, self.objective_vectors
                    )
                    self.__sprob_3.nadir = self.nadir
                    self.__sprob_3.ideal = self.ideal

                    self.__solver_3 = ASFScalarSolver(
                        self.__sprob_3, DiscreteMinimizer()
                    )
                    self.__solver_3.asf = PointMethodASF(
                        self.nadir, self.ideal
                    )

                res3 = self.__solver_3.solve(z_bar)  # type: ignore
                res_all_xs.append(res3[0])
                res_all_fs.append(res3[1][0])

            if self.n_points >= 4:
                # subproblem 4
                if self.__sprob_4 is None:
                    # create the subproblem and solver
                    self.__sprob_4 = ScalarDataProblem(
                        self.pareto_front, self.objective_vectors
                    )
                    self.__sprob_4.nadir = self.nadir
                    self.__sprob_4.ideal = self.ideal

                    self.__solver_4 = ASFScalarSolver(
                        self.__sprob_4, DiscreteMinimizer()
                    )
                    self.__solver_4.asf = AugmentedGuessASF(
                        self.nadir, self.ideal, self.__ind_set_free
                    )
                else:
                    # update the indices to be excluded in the existing
                    # solver's asf
                    self.__solver_4.asf.indx_to_exclude = (  # type: ignore
                        self.__ind_set_free
                    )

                res4 = self.__solver_4.solve(z_bar)  # type: ignore
                res_all_xs.append(res4[0])
                res_all_fs.append(res4[1][0])

        # return the obtained solutions and the archiveof existing solutions
        # deepcopy, beacuse we dont want to return a reference to the archive
        return (
            np.array(res_all_xs),
            np.array(res_all_fs),
            deepcopy(self.archive),
        )

    def interact(
        self,
        most_preferred_point: Optional[np.ndarray] = None,
        classifications: Optional[List[Tuple[str, Optional[float]]]] = None,
        n_generated_solutions: int = -1,
        save_points: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        search_between_points: Optional[Tuple[np.ndarray]] = None,
        n_intermediate_solutions: int = 1,
    ):
        """Handle the the preferneces given by the DM and set up variables for
        the next iterations

        Args:
            most_preferred_point (Optional[np.ndarray]): The most preferred
            point.
            classificaitons (Optional[List[Tuple[str, Optional[float]]]]): The
            classifications of the objective functions.
            n_generated_solutions (Optional[int]): The number of solutions to
            be generated in the next iteration.
            save_points (Optional[List[Tuple[np.ndarray, np.ndarray]]]): A list
            of points to be saved in the archive.
            search_between_points (Optional[Tuple[np.ndarray]]): The two points
            between intermediate solutions are searched for.
            n_intermediate_solutions (int): The number of intermediate
            solutions generated between the two given points.

        """
        if most_preferred_point is not None:
            self.current_point = most_preferred_point

        if classifications is not None:
            # parse new classificaitons and sort the indices and given
            # prefernece in an usable form
            self.classifications = classifications

        if n_generated_solutions > 0:
            # change the number of subrpoblems to be solved in the next
            # iteration
            self.n_points = n_generated_solutions

        if save_points is not None:
            # save the given points in the archive
            self.archive.extend(save_points)

        if search_between_points is not None:
            # set the flag indicating a preference to generate intermediate
            # points in the next iteration
            self.generate_intermediate = True
            self.search_between_points = search_between_points  # type: ignore
            self.n_intermediate_solutions = n_intermediate_solutions
