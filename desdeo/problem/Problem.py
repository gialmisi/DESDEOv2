"""Here various classes are defined that represent multiobjective optimization
problems.

"""

import logging
import logging.config
from abc import ABC, abstractmethod
from os import path
from typing import List, Optional, Tuple, Union

import numpy as np

from desdeo.problem.Constraint import ScalarConstraint
from desdeo.problem.Objective import ScalarObjective
from desdeo.problem.Variable import Variable

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class ProblemError(Exception):
    """Raised when an error related to the Problem class is encountered.

    """


class ProblemBase(ABC):
    """The base class from which every other class representing a problem should
    derive.  This class presents common interface for message broking in for
    the manager.

    """

    def __init__(self):
        self.__nadir: np.ndarray = None
        self.__ideal: np.ndarray = None
        self.__n_of_objectives: int = 0
        self.__n_of_variables: int = 0

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
    def n_of_objectives(self) -> int:
        return self.__n_of_objectives

    @n_of_objectives.setter
    def n_of_objectives(self, val: int):
        self.__n_of_objectives = val

    @property
    def n_of_variables(self) -> int:
        return self.__n_of_variables

    @n_of_variables.setter
    def n_of_variables(self, val: int):
        self.__n_of_variables = val

    @abstractmethod
    def get_variable_bounds(self):
        pass

    @abstractmethod
    def evaluate(
        self, decision_variables: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the problem using an ensemble of input vectors.

        Args:
            decision_variables (np.ndarray): An array of decision variable
            input vectors.

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
        objectives (List[ScalarObjective]): A list containing the objectives of
        the problem.
        constraints (List[ScalarConstraint]): A list conatining the constraints
        of the problem.

    """

    def __init__(
        self,
        objectives: List[ScalarObjective],
        variables: List[Variable],
        constraints: List[ScalarConstraint],
        nadir: Optional[np.ndarray] = None,
        ideal: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.__objectives: List[ScalarObjective] = objectives
        self.__variables: List[Variable] = variables
        self.__constraints: List[ScalarConstraint] = constraints

        self.n_of_objectives: int = len(self.objectives)
        self.n_of_variables: int = len(self.variables)

        if self.constraints is not None:
            self.__n_of_constraints: int = len(self.constraints)
        else:
            self.__n_of_constraints = 0

        # Nadir vector must be the same size as the number of objectives
        if nadir is not None:
            if len(nadir) != self.n_of_objectives:
                msg = (
                    "The length of the nadir vector does not match the"
                    "number of objectives: Length nadir {}, number of "
                    "objectives {}."
                ).format(len(nadir), self.n_of_objectives)
                logger.debug(msg)
                raise ProblemError(msg)

        # Ideal vector must be the same size as the number of objectives
        if ideal is not None:
            if len(ideal) != self.n_of_objectives:
                msg = (
                    "The length of the ideal vector does not match the"
                    "number of objectives: Length ideal {}, number of "
                    "objectives {}."
                ).format(len(ideal), self.n_of_objectives)
                logger.debug(msg)
                raise ProblemError(msg)

        # Nadir and ideal vectors must match in size
        if nadir is not None and ideal is not None:
            if len(nadir) != len(ideal):
                msg = (
                    "The length of the nadir and ideal point don't match:"
                    " length of nadir {}, length of ideal {}."
                ).format(len(nadir), len(ideal))
                logger.debug(msg)
                raise ProblemError(msg)

        self.nadir = nadir
        self.ideal = ideal

    @property
    def n_of_constraints(self) -> int:
        return self.__n_of_constraints

    @n_of_constraints.setter
    def n_of_constraints(self, val: int):
        self.__n_of_constraints = val

    @property
    def objectives(self) -> List[ScalarObjective]:
        return self.__objectives

    @objectives.setter
    def objectives(self, val: List[ScalarObjective]):
        self.__objectives = val

    @property
    def variables(self) -> List[Variable]:
        return self.__variables

    @variables.setter
    def variables(self, val: List[Variable]):
        self.__variables = val

    @property
    def constraints(self) -> List[ScalarConstraint]:
        return self.__constraints

    @constraints.setter
    def constraints(self, val: List[ScalarConstraint]):
        self.__constraints = val

    def get_variable_bounds(self) -> Union[np.ndarray, None]:
        """Return the upper and lower bounds of each decision variable present
        in the problem as a 2D numpy array. The first column corresponds to the
        lower bounds of each variable, and the second column to the upper
        bound.

        Returns:
           np.ndarray: Lower and upper bounds of each variable
           as a 2D numpy array. If undefined variables, return None instead.

        """
        if self.variables is not None:
            bounds = np.ndarray((self.n_of_variables, 2))
            for ind, var in enumerate(self.variables):
                bounds[ind] = np.array(var.get_bounds())
            return bounds
        else:
            logger.debug(
                "Attempted to get variable bounds for a "
                "ScalarMOProblem with no defined variables."
            )
            return None

    def get_variable_names(self) -> List[str]:
        """Return the variable names of the variables present in the problem in
        the order they were added.

        Returns:
            List[str]: Names of the variables in the order they were added.

        """
        return [var.name for var in self.variables]

    def get_objective_names(self) -> List[str]:
        """Return the names of the objectives present in the problem in the
        order they were added.

        Returns:
            List[str]: Names of the objectives in the order they were added.

        """
        return [obj.name for obj in self.objectives]

    def get_variable_lower_bounds(self) -> np.ndarray:
        """Return the lower bounds of each variable as a list. The order of the bounds
        follows the order the variables were added to the problem.

        Returns:
            np.ndarray: An array with the lower bounds of the variables.
        """
        return np.array([var.get_bounds()[0] for var in self.variables])

    def get_variable_upper_bounds(self) -> np.ndarray:
        """Return the upper bounds of each variable as a list. The order of the bounds
        follows the order the variables were added to the problem.

        Returns:
            np.ndarray: An array with the upper bounds of the variables.
        """
        return np.array([var.get_bounds()[1] for var in self.variables])

    def evaluate(
        self, decision_variables: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the problem using an ensemble of input vectors.

        Args:
            decision_variables (np.ndarray): An array of decision variable
            input vectors.

        Returns:
            (tuple): tuple containing:
                solutions (np.ndarray): The corresponding objective function
                values for each input vector.
                constraints (np.ndarray): The constraint values of the problem
                corresponding each input vector.

        """
        # Reshape decision_variabless with single row to work with the code
        shape = np.shape(decision_variables)
        if len(shape) == 1:
            decision_variables = np.reshape(decision_variables, (1, shape[0]))

        (n_rows, n_cols) = np.shape(decision_variables)

        if n_cols != self.n_of_variables:
            msg = (
                "The length of the input vectors does not match the number "
                "of variables in the problem: Input vector length {}, "
                "number of variables {}."
            ).format(n_cols, self.n_of_variables)
            logger.debug(msg)
            raise ProblemError(msg)

        objective_values: np.ndarray = np.ndarray(
            (n_rows, self.n_of_objectives), dtype=float
        )
        if self.__n_of_constraints > 0:
            constraint_values: np.ndarray = np.ndarray(
                (n_rows, self.__n_of_constraints), dtype=float
            )
        else:
            constraint_values = None

        # Calculate the objective values
        for (col_i, objective) in enumerate(self.objectives):
            objective_values[:, col_i] = np.array(
                list(map(objective.evaluate, decision_variables))
            )

        # Calculate the constraint values
        if constraint_values is not None:
            for (col_i, constraint) in enumerate(self.constraints):
                constraint_values[:, col_i] = np.array(
                    list(
                        map(
                            constraint.evaluate,
                            decision_variables,
                            objective_values,
                        )
                    )
                )

        return (objective_values, constraint_values)
