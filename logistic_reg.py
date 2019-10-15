import numpy as np
import random
import matplotlib.pyplot as plt
from math import exp as e

random.seed(42)
np.random.seed(42)

a = 0.8
b = 20.

def f(x):
    return a*x + b

def z(t0, t1, t2, x1, x2):
	return t0 + t1*x1 + t2*x2

def sig(z):
	return 1/(1 + e(-z))

m = 200
X1 = np.random.randint(1000, size=m).astype(dtype=np.float32)
X2 = np.random.randint(1000, size=m).astype(dtype=np.float32)
Y = np.array([1 if x2 > f(x1) else 0
for x1, x2 in zip(X1, X2)], dtype=np.float32)

for i in range(m):
	if random.random() < 0.03:
		Y[i] = 1 - Y[i] 

t0, t1, t2 = 0, 0, 0
lr = 0.001

for _ in range(10000):
	dt0 , dt1, dt2 = 0, 0, 0
	for i in range(m):
		dz = sig(z(t0, t1, t2, X1[i], X2[i])) - Y[i]

		dt0 += dz
		dt1 += dz*X1[i]
		dt2 += dz*X2[i]

	dt0 /= m
	dt1 /= m
	dt2 /= m  
  
	t0 -= lr*dt0
	t1 -= lr*dt1
	t2 -= lr*dt2

_Y = [(-t0 - t1*xx) / t2 for xx in X1]

red = np.where(Y==1)
blue = np.where(Y==0)
plt.plot(X1[red], X2[red], 'ro')
plt.plot(X1[blue], X2[blue], 'bo')
plt.plot(X1, _Y, 'y')
plt.show()
