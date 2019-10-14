import numpy as np
import random
from matplotlib import pyplot as plt

m = 0.3
b = 50.


def f(x):
    return m*x + b

def h(t0, t1, x):
	return t0 + t1*x

X = np.random.randint(1000, size=50).astype(dtype=np.float32)
Y = np.array([f(i) + 10 * (random.random()-0.5) for i in X], dtype=np.float32)

t0, t1 = 0.0, 0.0
lr = 0.000001
j = []

for _ in range(1000):
	dt0, dt1 = 0, 0
	dj = 0
	for i in range(len(X)):
		diff = h(t0, t1, X[i]) - Y[i]
		dt0 += diff
		dt1 += diff * X[i]
		dj += diff ** 2

	print(dj * 2 / len(X))
	j.append(dj * 2 / len(X))

	t0 -= lr * dt0 / len(X)
	t1 -= lr * dt1 / len(X)


_Y = [h(t0, t1, x) for x in X]

plt.plot(X, _Y)
plt.plot(X, Y, 'ro')
plt.show()

_x = np.arange(1000)
plt.plot(_x, j)
plt.show()

