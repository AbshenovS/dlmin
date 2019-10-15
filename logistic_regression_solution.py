import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

a = 0.6
b = 20.

def f(x):
    return a*x + b

def sigma(z):
	return 1/(1 + np.exp(-z))

m = 100
X1 = np.random.randint(1000, size=m).astype(dtype=np.float32)
X2 = np.random.randint(1000, size=m).astype(dtype=np.float32)
Y = np.array([1 if x2 > f(x1) else 0
			for x1, x2 in zip(X1, X2)], dtype=np.float32)
for i in range(m):
	if random.random() < 0.03:
		Y[i] = 1 - Y[i] 

X = np.stack([np.ones(m), X1, X2]).T
theta = np.array([0., 0., 0.])
lr = 1.

dw_s=0
for _ in range(100000):
	Z = X.dot(theta)
	a = sigma(Z)
	dz = a - Y
	dw = X.T.dot(dz)/m
	dw_s = dw_s*0.9 + 0.1*dw
	theta -= lr*dw_s

print(theta)
aa = -theta[1]/theta[2]
bb = -theta[0]/theta[2]
def g(x):
	return aa*x + bb

red = np.where(Y==1)
blue = np.where(Y==0)
plt.plot(X1[red], X2[red], 'ro')
plt.plot(X1[blue], X2[blue], 'bo')

_x = [-100, 1100]
_y = [g(i) for i in _x]
plt.plot(_x, _y)

plt.show()
