import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x):
	return 10 - 5*x[0] + 2*x[1]

def h(theta, x):
	return theta[0] + theta[1]*x[0] + theta[2]*x[1]

X = np.random.randint(500, size=(100, 2)).astype(dtype=np.float32)
Y = np.array([f(i) + 100 * (random.random()-0.5) for i in X])

X = np.append(np.ones((100, 1)), X, axis=1)

theta = np.array([0., 0., 0.])
lr = 0.000001

for _ in range(1000):
	a = X.dot(theta.T)
	dt = X.T.dot(a - Y)/len(X)
	theta = theta - lr*dt

print(theta)

_x = range(-100, 600, 100)
_y = range(-100, 600, 100)
_x, _y = np.meshgrid(_x, _y)
z = np.array([h(theta, i) for i in zip(_x, _y)])

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(_x, _y, z, alpha=0.2)

ax = plt.gca()
ax.scatter(X[:,1], X[:,2], Y, c='m')

plt.show()

