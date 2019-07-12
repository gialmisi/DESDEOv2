"""Various interactive multi-objective optimization methods belonging to the
NAUTILUS-family are defined here.

"""

import logging
import logging.config
from os import path
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np

from desdeo.methods.InteractiveMethod import (
    InteractiveMethodBase,
    InteractiveMethodError,
)
from desdeo.problem.Problem import ProblemBase
from desdeo.solver.PointSolver import IdealAndNadirPointSolver

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
        self.__n_of_iterations: int = 0
        self.__current_iteration: int = 1
        self.__lower_bound: np.ndarray = None
        self.__upper_bound: np.ndarray = None

    @property
    def n_of_iterations(self) -> int:
        return self.__n_of_iterations

    @n_of_iterations.setter
    def n_of_iterations(self, val: int):
        self.__n_of_iterations = val

    @property
    def initialization_requirements(self) -> List[Tuple[str, Type, Callable]]:
        return self.__initialization_requirements

    def initialize(self, initialization_parameters: Dict[str, Any]):
        # Parse and set the given initialization parameters
        for (key, t, setter) in self.__initialization_requirements:
            if key in initialization_parameters:
                if isinstance(initialization_parameters[key], t):
                    # Paramter found and is of correct type
                    setter(self, initialization_parameters[key])
                else:
                    # Wrong type of paramter
                    msg = (
                        "The type '{}' of the supplied initialization "
                        "parameter '{}' does not match the expected "
                        "type'{}'."
                    ).format(
                        str(type(initialization_parameters[key])), key, str(t)
                    )
                    logger.debug(msg)
                    raise InteractiveMethodError(msg)
            else:
                # Missing parameter
                msg = ("Missing initialization parameter '{}'").format(key)
                logger.debug(msg)
                raise InteractiveMethodError(msg)

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

        self.__lower_bound = self.problem.ideal
        self.__upper_bound = self.problem.nadir

    def iterate(self):
        pass

    def interact(self):
        pass
