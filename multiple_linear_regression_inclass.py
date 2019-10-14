import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x):
	return 10 - 5*x[0] + 2*x[1]

def h(theta0, theta1, theta2, x):
	return theta0 + theta1*x[0] + theta2*x[1]

X = np.random.randint(500, size=(100, 2)).astype(dtype=np.float32)
Y = np.array([f(i) + 100 * (random.random()-0.5) for i in X])


theta0 = 0.0
theta1 = 0.0
theta2 = 0.0
lr = 0.000001

for _ in range(1000):
	dt0 = 0
	dt1 = 0
	dt2 = 0
	for i in range(len(X)):
		diff = h(theta0, theta1, theta2, X[i]) - Y[i]
		dt0 += diff
		dt1 += diff*X[i][0]
		dt2 += diff*X[i][1]

	dt0 /= len(X)
	dt1 /= len(X)
	dt2 /= len(X)  
  
	theta0 -= lr*dt0
	theta1 -= lr*dt1
	theta2 -= lr*dt2

print(theta0, theta1, theta2)

x = range(-100, 600, 100)
y = range(-100, 600, 100)
x, y = np.meshgrid(x, y)
z = np.array([h(theta0, theta1, theta2, i) for i in zip(x, y)])

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(x, y, z, alpha=0.2)

ax = plt.gca()
ax.scatter(X[:,0], X[:,1], Y, c='m')

plt.show()	
