"""Define numerical methods to be used in the solvers.
"""
import logging
import logging.config
from abc import ABC, abstractmethod
from os import path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import differential_evolution

log_conf_path = path.join(
    path.dirname(path.abspath(__file__)), "../logger.cfg"
)
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)


class NumericalMethodError(Exception):
    """Raised when an error is encountered in the numerical method classes.

    """


class NumericalMethodBase(ABC):
    """Define an abstract class for all methods to follow

    Args:
        method(Callable[[Any], np.ndarray]): A callable method that minimizes a
            given (set of) funtions and returns the ideal solution(s).

    Attributes:
        method(Callable[[Any], np.ndarray]): A callable method that minimizes a
        given (set of) funtions and returns the ideal solution(s).

    """

    def __init__(self, method: Callable[[Any], np.ndarray]):
        self.__method = method

    @property
    def method(self) -> Any:
        return self.__method

    @method.setter
    def method(self, val: Any):
        self.__method = val

    @abstractmethod
    def run(
        self,
        evaluator: Callable[..., np.ndarray],
        bounds: np.ndarray,
        evaluator_args: Union[Dict[str, Any], Tuple[Any]] = {},
        variables: Optional[np.ndarray] = None,
        objectives: Optional[np.ndarray] = None,
    ):
        """Passes the appropiate parameters to the underlying method and
        returns the solution given by the method.

        Args:
            evaluator(Callable[[np.ndarray, Any], np.ndarray]):
            A function to be evaluated and minimized.
            bounds(np.ndarray): The bounds of the variables as a 2D array with
            each row representing the lower (first column) and upper (last
            column) bounds of each variable.
            evaluator_args(Optional[Dict[str, Any]]): An optional tuple
            containing positional arguments to be passed to the evaluator.

        Returns:
            The solution computed by the numerical method.

        """
        pass


class ScipyDE(NumericalMethodBase):
    """Uses the differential solver implemented in SciPy.

    Attributes:
        method_kwargs(Optional[Any]): The keyword arguments to be passed to the
        differential evolution algorithm implemented in SciPy, see the SciPy
        documentation for more info.

    Args:
        method_kwargs(Optional[Any]): The keyword arguments to be passed to the
        differential evolution algorithm implemented in SciPy, see the SciPy
        documentation for more info.

    """

    def __init__(
        self,
        method_kwargs: Optional[Any] = {
            "tol": 0.000001,
            "popsize": 10,
            "maxiter": 50000,
            "polish": True,
        },
    ):
        super().__init__(differential_evolution)
        self.method_kwargs = method_kwargs

    @property
    def method_kwargs(self) -> Any:
        return self.__method_kwargs

    @method_kwargs.setter
    def method_kwargs(self, val: Any):
        self.__method_kwargs = val

    def run(
        self,
        evaluator: Callable[[np.ndarray, Any], np.ndarray],
        bounds: np.ndarray,
        evaluator_args: Union[Dict[str, Any], Tuple[Any]] = {},
        variables: Optional[np.ndarray] = None,
        objectives: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run the differential solver minimizinf the given evaluator and
        following the given variable bounds.

        Args:
            evaluator(Callable[[np.ndarray, Any], np.ndarray]):
            A function to be evaluated and minimized.
            bounds(np.ndarray): The bounds of the variables as a 2D array with
            each row representing the lower (first column) and upper (last
            column) bounds of each variable.
            evaluator_args(Union[Dict[str, Any], Tuple[Any], None]): An
            optional tuple containing positional arguments to be passed to the
            evaluator.

        Returns:
            np.ndarray: An array containing the optimal solution reached by the
            differential evolution method.

        Raises:
            NumericalMethodError: Something goes wrong with the evaluator or
            the minimization.

        """
        if evaluator_args is None:
            results = self.method(evaluator, bounds, **self.method_kwargs)
        else:
            results = self.method(
                evaluator, bounds, args=evaluator_args, **self.method_kwargs
            )

        if results.success:
            return results.x
        else:
            msg = (
                "The differential solver was not successful. " "Reason: {}"
            ).format(results.message)
            logger.debug(msg)
            raise NumericalMethodError(msg)


class DiscreteMinimizer(NumericalMethodBase):
    """Finds the minimum value from a discrete set of pre-defined points.

    """

    def __init__(self):
        super().__init__(self.minimizer)

    def minimizer(
        self,
        evaluator: Callable[..., np.ndarray],
        bounds: np.ndarray,
        variables: Optional[np.ndarray] = None,
        objectives: Optional[np.ndarray] = None,
        kwargs: Dict[str, Any] = {},
    ) -> np.ndarray:
        """
        Args:
            evaluator (Callable[[np.ndarray, Any], np.ndarray]):
            A function to be evaluated and minimized.
            bounds (np.ndarray): The bounds of the variables as a 2D array with
            each row representing the lower (first column) and upper (last
            column) bounds of each variable.
            variables (Optional[np.ndarray]): 2D array with the variables of
            the discrete problem.
            objectives (Optional[np.ndarray]): 2D array with the objective
            function values of the discrete problem.
            kwargs (Optional[Dict[str, Any]]): Extra arguments to be passed to
            the evaluator.

        Returns:
            np.ndarray: The variables that results in the objective values that
            minimize the evaluator.

        Raises:
            NumericalMethodError: variables or objectives not specified.

        """
        if variables is None:
            msg = "Variables must be specified for the minimizer to work."
            logger.error(msg)
            raise NumericalMethodError(msg)
        if objectives is None:
            msg = "Objectives must be specified for the minimizer to work."
            logger.error(msg)
            raise NumericalMethodError(msg)

        if bounds is not None:
            mask_lower_bounds = np.all(
                np.greater(variables, bounds[:, 0]), axis=1
            )
            mask_upper_bounds = np.all(
                np.less(variables, bounds[:, 1]), axis=1
            )
            mask_feasible = np.logical_and(
                mask_lower_bounds, mask_upper_bounds
            )

            feasible_objectives = objectives[mask_feasible]
            feasible_variables = variables[mask_feasible]
        else:
            feasible_objectives = objectives
            feasible_variables = variables

        res = evaluator(feasible_variables, feasible_objectives, **kwargs)
        idx = np.argmin(res)

        return feasible_variables[idx]

    def run(
        self,
        evaluator: Callable[[np.ndarray, Any], np.ndarray],
        bounds: np.ndarray,
        evaluator_args: Union[Dict[str, Any], Tuple[Any]] = {},
        variables: Optional[np.ndarray] = None,
        objectives: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run the minimizer minimizing the given evaluator and
        following the given variable bounds.

        Args:
            evaluator(Callable[[np.ndarray, Any], np.ndarray]):
            A function to be evaluated and minimized.
            bounds(np.ndarray): The bounds of the variables as a 2D array with
            each row representing the lower (first column) and upper (last
            column) bounds of each variable.
            evaluator_args(Union[Dict[str, Any], Tuple[Any], None]): An
            optional tuple containing positional arguments to be passed to the
            evaluator.

        Returns:
            np.ndarray: An array containing the optimal solution reached by the
            differential evolution method.

        Raises:
            NumericalMethodError: Something goes wrong with the evaluator or
            the minimization.

        """
        results = self.method(
            evaluator,
            bounds,
            variables=variables,
            objectives=objectives,
            kwargs=evaluator_args,
        )

        return results
