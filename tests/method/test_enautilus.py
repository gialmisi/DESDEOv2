import numpy as np
import pytest
from pytest import approx
import os

from desdeo.methods.InteractiveMethod import InteractiveMethodError
from desdeo.methods.Nautilus import ENautilus


@pytest.fixture
def sphere_pareto():
    """Return a tuple of points representing the angle parametrized surface of
    a sphere's positive octant.  The first element represents the theta and phi
    angle values and the second the corresponding cartesian (x,y,z)
    coordinates

    """
    dirname = os.path.dirname(__file__)
    relative = "../../data/pareto_front_3d_sphere_1st_octant_surface.dat"
    filename = os.path.join(dirname, relative)
    p = np.loadtxt(filename)
    return (p[:, :2], p[:, 2:])


def test_initialization(sphere_pareto):
    method = ENautilus()
    x, f = sphere_pareto
    method.initialize(x, f)
