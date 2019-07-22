import numpy as np
import pytest

from desdeo.methods.Nautilus import Nautilus


@pytest.fixture
def NautilusInitializedRiver(RiverPollutionProblem):
    problem = RiverPollutionProblem
    problem.nadir_point = np.array([-4.07, -2.83, -0.32, 9.71])
    problem.ideal_point = np.array([-6.34, -3.44, -7.50, 0.00])
    method = Nautilus(problem)
    method.initialize()

    return method
