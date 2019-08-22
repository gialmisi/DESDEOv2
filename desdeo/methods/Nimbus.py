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

from typing import Optional, List, Tuple


log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


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
        self.__classifications: List[Tuple[str, Optional[float]]]
        self.__available_classifications: List[str] = [
            "<",
            "<=",
            "=",
            ">=",
            "0",
        ]
        # nadir
        self.__nadir: np.ndarray = None
        # ideal
        self.__ideal: np.ndarray = None
        # current index of the solution used
        self.__cind: np.int = None
        # starting objective vector
        self.__start: np.ndarray = None
        # solution archive
        self.__archive: List[np.ndarray] = []
        # number of point to be generated
        self.__n_points: int = 0

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
    def start(self) -> np.ndarray:
        return self.__star

    @start.setter
    def start(self, val: np.ndarray):
        """Set the starting point for the algorithm. The dimensions of the
        starting point must match the dimensions of the row vectors in
        objective_vectors.

        Parameters:
            val(np.ndarray): The starting point.

        Raises:
            InteractiveMethodError: The staring point's dimension does not
            match that of the given pareto optimal objective vectors.

        """
        if len(val) != self.objective_vectors.shape[1]:
            msg = (
                "Starting point dimensions '{}' don't match the objective"
                " vector dimensions '{}'"
            ).format(len(val), self.objective_vectors.shape[1])
            logger.error(msg)
            raise InteractiveMethodError(msg)

        self.__star = val

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
        if val < 0 or val > 4:
            msg = ("The given number '{}' of solutions to be generated is not"
                   " between 1 and 4 (inclusive)").format(val)
            logger.error(msg)
            raise InteractiveMethodError(msg)
        self.__n_points = val

    def initialize(
        self,
        n_solutions: int,
        starting_point: Optional[np.ndarray] = None,
        pareto_front: Optional[np.ndarray] = None,
        objective_vectors: Optional[np.ndarray] = None,
    ):
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

        # if the starting point has been specified, use that. Otherwise, use
        # random one.
        if starting_point is not None:
            self.start = starting_point

    def iterate(self):
        pass

    def interact(self):
        pass
