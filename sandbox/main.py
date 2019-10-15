"""This is for purely testing.

"""
from desdeov2.methods.Nautilus import ENautilusB
from desdeov2.problem.Problem import ScalarDataProblem

import numpy as np

data = np.loadtxt("./data/pareto_front_3d_sphere_1st_octant_surface.dat")
problem = ScalarDataProblem(data[:, 0:2], data[:, 2:])
method = ENautilusB(problem)

method.initialize(10, 10)
limits, dist = method.iterate()

method.nadir = np.array([2, 2, 2])

new_point = np.array([0, 0, 1])

method.interact(new_point)
limits, dist = method.iterate()
print(limits)
print(dist)

