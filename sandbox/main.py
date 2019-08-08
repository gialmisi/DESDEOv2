"""This is for purely testing.

"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

a = np.array([[np.cos(th), np.sin(th)] for th in np.linspace(0, np.pi/2, 100)])

kmeans = KMeans(n_clusters=8)
kmeans.fit(a)

c = kmeans.cluster_centers_

plt.scatter(a[:, 0], a[:, 1], c='b', alpha=0.1)
plt.scatter(c[:, 0], c[:, 1], c='r')

plt.show()
