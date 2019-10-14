import numpy as np
import random
import matplotlib.pyplot as plt

m = 0.3
b = 500.

def f(x):
    return m*x + b

def h(theta0, theta1, x):
	return theta0 + theta1*x

X = np.random.randint(1000, size=50).astype(dtype=np.float32)
Y = np.array([f(i) + 21 * (random.random()-0.5) for i in X], dtype=np.float32)

theta0 = 0.0
theta1 = 0.0
lr = 0.01

v0, v1 = 0.0, 0.0
beta = 0.999

for _ in range(10000):
	dt0 = 0
	dt1 = 0
	for i in range(len(X)):
		diff = h(theta0, theta1, X[i]) - Y[i]
		dt0 += diff
		dt1 += diff*X[i]

	dt0 /= len(X)
	dt1 /= len(X)

	v0 = beta * v0 + (1 - beta) * dt0
	v1 = beta * v1 + (1 - beta) * dt1

	theta0 -= lr*v0
	theta1 -= lr*v1

print(theta0, theta1)

_x = [0, 1000]
_y = [h(theta0, theta1, i) for i in _x]

plt.plot(X, Y, 'ro')
plt.plot(_x, _y)
plt.show()
