"""This is for purely testing.

"""
from desdeo.problem.Problem import ScalarMOProblem
from desdeo.problem.Variable import Variable
from desdeo.problem.Objective import ScalarObjective
from desdeo.solver.ScalarSolver import ASFScalarSolver
from desdeo.solver.PointSolver import IdealAndNadirPointSolver

from desdeo.solver.ASF import ReferencePointASF

from desdeo.solver.NumericalMethods import ScipyDE

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def RiverPollutionProblem():
    """Return the river pollution problem as defined in `Miettinen 2010`_

    .. _Miettinen 2010:
        Miettinen, K.; Eskelinen, P.; Ruiz, F. & Luque, M.
        NAUTILUS method: An interactive technique in multiobjective
        optimization based on the nadir point
        Europen Joural of Operational Research, 2010, 206, 426-434

    """
    # Variables
    variables = []
    variables.append(Variable("x_1", 0.5, 0.3, 1.0))
    variables.append(Variable("x_2", 0.5, 0.3, 1.0))

    # Objectives
    objectives = []
    objectives.append(ScalarObjective("f_1", lambda x: -4.07 - 2.27 * x[0]))

    objectives.append(
        ScalarObjective(
            "f_2",
            lambda x: -2.60
            - 0.03 * x[0]
            - 0.02 * x[1]
            - 0.01 / (1.39 - x[0] ** 2)
            - 0.30 / (1.39 - x[1] ** 2),
        )
    )

    objectives.append(
        ScalarObjective("f_3", lambda x: -8.21 + 0.71 / (1.09 - x[0] ** 2))
    )

    objectives.append(
        ScalarObjective("f_4", lambda x: -0.96 + 0.96 / (1.09 - x[1] ** 2))
    )

    # problem
    problem = ScalarMOProblem(objectives, variables, [])

    return problem


# problem = RiverPollutionProblem()
# solver = ASFScalarSolver(problem, ScipyDE())
# idealnadir = IdealAndNadirPointSolver(problem, ScipyDE())
# ideal, nadir = idealnadir.solve()
# solver.asf = ReferencePointASF(
#     np.array([0.25, 0.25, 0.25, 0.25]), nadir, ideal - 1e-6, rho=1e-6
# )

# n = 500
# refs = np.random.uniform(ideal, nadir, (n, 4))

# xs = np.zeros((n, 2))
# fs = np.zeros((n, 4))
# for (i, r) in enumerate(refs):
#     xs[i, :], (fs[i, :], _) = solver.solve(r)

# X = np.hstack((xs, fs))
# np.savetxt(
#     "riverpollution.dat",
#     X,
#     header=(
#         "Pareto representation for the river pollution problem.\nDecision "
#         "variable values are listen on each row first, then the corresponding "
#         "objective function values\nNumber of decision variables: 2\nNumber of"
#         " objective functions: 4"
#     ),
# )

data = np.loadtxt("/home/kilo/workspace/DESDEOv2/riverpollution.dat")
xs, fs = data[:, 0:2], data[:, 2:]


plt.scatter(xs[:, 0], xs[:, 1])
plt.show()
