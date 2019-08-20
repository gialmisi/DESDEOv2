"""Define numerical methods to be used in the solvers.
"""
import logging
import logging.config
from abc import ABC, abstractmethod
from os import path
from typing import Any, Callable, Optional, Tuple

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
        evaluator: Callable[[np.ndarray, Any], np.ndarray],
        bounds: np.ndarray,
        evaluator_args: Optional[Tuple[Any]] = None,
    ) -> np.ndarray:
        """Passes the appropiate parameters to the underlying method and
        returns the solution given by the method.

        Parameters:
            evaluator(Callable[[np.ndarray, Any], np.ndarray]):
            A function to be evaluated and minimized.
            bounds(np.ndarray): The bounds of the variables as a 2D array with
            each row representing the lower (first column) and upper (last
            column) bounds of each variable.
            evaluator_args(Optional[Tuple[Any]]): An optional tuple containing
            positional arguments to be passed to the evaluator.

        Returns:
            The solution evaluated by the numerical method.

        """
        pass


class ScipyDE(NumericalMethodBase):
    """Uses the differential solver implemented in SciPy.

    Attributes:
    method_kwargs(Optional[Any]): The keyword arguments to be passed to the
    numerical method.

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
        evaluator_args: Optional[Tuple[Any]] = None,
    ) -> np.ndarray:
        """Run the differential solver minimizinf the given evaluator and
        following the given variable bounds.

        Parameters:
            evaluator(Callable[[np.ndarray, Any], np.ndarray]):
            A function to be evaluated and minimized.
            bounds(np.ndarray): The bounds of the variables as a 2D array with
            each row representing the lower (first column) and upper (last
            column) bounds of each variable.
            evaluator_args(Optional[Tuple[Any]]): An optional tuple containing
            positional arguments to be passed to the evaluator.

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
