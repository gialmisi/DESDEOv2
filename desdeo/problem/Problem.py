"""Here various classes are defined that represent multiobjective optimization
problems.

"""

import logging
import logging.config
from os import path
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


import numpy as np
from desdeo.problem.Variable import Variable
from desdeo.problem.Objective import ScalarObjective
from desdeo.problem.Constraint import ScalarConstraint

log_conf_path = path.join(path.dirname(path.abspath(__file__)),
                          "../../logger.cfg")
logging.config.fileConfig(fname=log_conf_path,
                          disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class ProblemError(Exception):
    """Raised when an error related to the Problem class is encountered.

    """


class ProblemBase(ABC):
    """The base class from which every other class representing a problem should
    derive.  This class presents common interface for message broking in for
    the manager.

    """

    @abstractmethod
    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluates the problem using an ensemble of input vectors.

        Args:
            population (np.ndarray): An array of decision variable input
            vectors.

        Returns:
            (tuple): tuple containing:
                solutions (np.ndarray): The corresponding objective function
                values for each input vector.
                constraints (np.ndarray): The constraint values of the problem
                corresponding each input vector.

        """
        pass


class ScalarMOProblem(ProblemBase):
    """A multiobjective optimization problem with user defined objective funcitons,
    constraints and variables. The objectives each return a single scalar.

    Args:
        objectives (List[ScalarObjective]): A list containing the objectives of
        the problem.
        variables (List[Variable]): A list containing the variables of the
        problem.
        constraints (List[ScalarConstraint]): A list containing the
        constraints of the problem. If no constraints exist, None may
        be supllied as the value.
        nadir (Optional[np.ndarray]): The nadir point of the problem.
        ideal (Optional[np.ndarray]): The ideal point of the problem.

    Attributes:
        n_of_objectives (int): The number of objectives in the problem.
        n_of_variables (int): The number of variables in the problem.
        n_of_constraints (int): The number of constraints in the problem.
        nadir (np.ndarray): The nadir point of the problem.
        ideal (np.ndarray): The ideal point of the problem.

    """
    def __init__(self,
                 objectives: List[ScalarObjective],
                 variables: List[Variable],
                 constraints: List[ScalarConstraint],
                 nadir: Optional[np.ndarray] = None,
                 ideal: Optional[np.ndarray] = None) -> None:
        self.__objectives: List[ScalarObjective] = objectives
        self.__variables: List[Variable] = variables
        self.__constraints: List[ScalarConstraint] = constraints

        self.__n_of_objectives: int = len(self.__objectives)
        self.__n_of_variables: int = len(self.__variables)
        if self.__constraints is not None:
            self.__n_of_constraints: int = len(self.__constraints)
        else:
            self.__n_of_constraints = 0

        # Nadir vector must be the same size as the number of objectives
        if nadir is not None:
            if len(nadir) != self.__n_of_objectives:
                msg = ("The length of the nadir vector does not match the"
                       "number of objectives: Length nadir {}, number of "
                       "objectives {}.").format(
                           len(nadir),
                           self.__n_of_objectives)
                logger.debug(msg)
                raise ProblemError(msg)

        # Ideal vector must be the same size as the number of objectives
        if ideal is not None:
            if len(ideal) != self.__n_of_objectives:
                msg = ("The length of the ideal vector does not match the"
                       "number of objectives: Length ideal {}, number of "
                       "objectives {}.").format(
                           len(ideal),
                           self.__n_of_objectives)
                logger.debug(msg)
                raise ProblemError(msg)

        # Nadir and ideal vectors must match in size
        if nadir is not None and ideal is not None:
            if len(nadir) != len(ideal):
                msg = ("The length of the nadir and ideal point don't match:"
                       " length of nadir {}, length of ideal {}.").format(
                         len(nadir),
                         len(ideal))
                logger.debug(msg)
                raise ProblemError(msg)

        self.__nadir = nadir
        self.__ideal = ideal

    @property
    def n_of_objectives(self) -> int:
        return self.__n_of_objectives

    @property
    def n_of_variables(self) -> int:
        return self.__n_of_variables

    @property
    def n_of_constraints(self) -> int:
        return self.__n_of_constraints

    @property
    def nadir(self) -> float:
        return self.__nadir

    @property
    def ideal(self) -> float:
        return self.__ideal

    def evaluate(self, population: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the problem using an ensemble of input vectors.

        Args:
            population (np.ndarray): An array of decision variable input
            vectors.

        Returns:
            (tuple): tuple containing:
                solutions (np.ndarray): The corresponding objective function
                values for each input vector.
                constraints (np.ndarray): The constraint values of the problem
                corresponding each input vector.

        """
        (n_rows, n_cols) = np.shape(population)
        if n_cols != self.__n_of_variables:
            msg = ("The length of the input vectors does not match the number "
                   "of variables in the problem: Input vector length {}, "
                   "number of variables {}.").format(
                       n_cols,
                       self.__n_of_variables)
            logger.debug(msg)
            raise ProblemError(msg)

        objective_values: np.ndarray = np.ndarray((n_rows,
                                                   self.__n_of_objectives),
                                                  dtype=float)
        if self.__n_of_constraints > 0:
            constraint_values: np.ndarray = \
                np.ndarray((n_rows,
                            self.__n_of_constraints),
                           dtype=float)
        else:
            constraint_values = None

        # Calculate the objective values
        for (col_i, objective) in enumerate(self.__objectives):
            objective_values[:, col_i] = \
                np.array(list(map(
                    objective.evaluate, population)))

        # Calculate the constraint values
        if constraint_values is not None:
            for (col_i, constraint) in enumerate(self.__constraints):
                constraint_values[:, col_i] = \
                    np.array(list(map(
                        constraint.evaluate, population, objective_values)))

        return (objective_values, constraint_values)
