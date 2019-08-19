"""This is for purely testing.

"""

from desdeo.solver.ASF import ReferencePointASF
import numpy as np

objectives = np.zeros((10, 5))
reference = np.ones(5)
preference = np.full(5, 2)

# print(objectives)
# print(reference)

# print(np.max(preference * (objectives - reference), axis=1))

print(0.5 * (np.sum((objectives - reference) / (np.zeros(5) - np.ones(5)), axis=-1)))

asf = ReferencePointASF(np.ones(5), np.ones(5), np.ones(5)-1)

res = asf(objectives, reference)
