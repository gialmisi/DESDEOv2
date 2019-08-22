"""Various interactive multiobjective optimization methods belonging to them
NIMBUS-family are defined here

"""

import logging
import logging.config
from os import path

import numpy as np

from desdeo.methods.InteractiveMethod import (
    InteractiveMethodBase,
    InteractiveMethodError,
)

from desdeo.problem.Problem import (
    ProblemBase,
    ScalarMOProblem,
    ScalarDataProblem,
)

from desdeo.utils.frozen import frozen


from typing import Optional, List, Tuple


log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


@frozen(logger)
class SNimbus(InteractiveMethodBase):
    """Implements the synchronous NIMBUS variant as defined in
       `Miettinen 2016`_

    .. _Miettinen 2006:
        Mietttinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922

    """

    def __init__(self, problem: Optional[ProblemBase] = None):
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
        self.__aspiration_levels: List[Tuple[int, float]] = []
        self.__upper_bounds: List[Tuple[int, float]] = []
        # nadir
        self.__nadir: np.ndarray = None
        # ideal
        self.__ideal: np.ndarray = None
        # current index of the solution used
        self.__cind: np.int = None
        # starting objective vector
        self.__current_point: np.ndarray = None
        # solution archive
        self.__archive: List[np.ndarray] = []
        # number of point to be generated
        self.__n_points: int = 0
        # flag to represent if first iteration
        self.__first_iteration = True

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
    def archive(self) -> List[np.ndarray]:
        return self.__archive

    @archive.setter
    def archive(self, val: List[np.ndarray]):
        self.__archive = val

    @property
    def current_point(self) -> np.ndarray:
        return self.__current_point

    @current_point.setter
    def current_point(self, val: np.ndarray):
        """Set the current point for the algorithm. The dimensions of the
        current point must match the dimensions of the row vectors in
        objective_vectors.

        Parameters:
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

        Parameters:
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
        self.__aspiration_levels = []
        self.__upper_bounds = []
        for (ind, cls) in enumerate(self.classifications):
            if cls[0] == "<":
                self.__ind_set_lt.append(ind)
                print(self.__ind_set_lt)
            elif cls[0] == "<=":
                self.__ind_set_lte.append(ind)
                self.__aspiration_levels.append((ind, cls[1]))
            elif cls[0] == "=":
                self.__ind_set_eq.append(ind)
            elif cls[0] == ">=":
                self.__ind_set_gte.append(ind)
                self.__upper_bounds.append((ind, cls[1]))
            elif cls[0] == "0":
                self.__ind_set_free.append(ind)
            else:
                msg = (
                    "Check that the classification '{}' is correct."
                ).format(cls)
                logger.error(msg)
                raise InteractiveMethodError(msg)

    def initialize(
        self,
        n_solutions: int,
        starting_point: Optional[np.ndarray] = None,
        pareto_front: Optional[np.ndarray] = None,
        objective_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Initialize the method and return the starting objective vector.

        Parameters:
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

        Note:
            The data to be used must be available in the underlying problem OR
            given explicitly.

        """
        self.n_points = n_solutions

        if isinstance(self.problem, ScalarDataProblem):
            self.pareto_front = self.problem.variables
            self.objective_vectors = self.problem.objectives

        elif pareto_front is not None and objective_vectors is not None:
            self.pareto_front = pareto_front
            self.objective_vectors = objective_vectors

        else:
            msg = (
                "Either a pre computed problem must be defined in this "
                "class or the pareto front and objective vectors must "
                "be explicitly given."
            )
            logger.error(msg)
            raise InteractiveMethodError(msg)

        self.nadir = np.max(self.objective_vectors, axis=0)
        self.ideal = np.min(self.objective_vectors, axis=0)

        self.__classifications = [None] * self.objective_vectors.shape[1]

        # if the starting point has been specified, use that. Otherwise, use
        # random one.
        if starting_point is not None:
            self.current_point = starting_point

        else:
            rind = np.random.randint(0, len(self.objective_vectors))
            self.current_point = self.objective_vectors[rind]

        return self.current_point

    def iterate(self):
        # if first iteration, just return the starting point
        if self.first_iteration:
            self.first_iteration = False
            return self.current_point

    def interact(self, classifications: List[Tuple[str, Optional[float]]]):
        self.classifications = classifications
        pass
