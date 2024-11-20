from Tikhonov import *
import numpy as np


f = lambda x: np.sin(x) + 2*np.cos(4 * x)
x = np.linspace(-np.pi, np.pi, 10)
y = f(x)

M = GenMTrigBasis(x, 4)

print(lowPassPenaltyMatrix(3, 1, 1, 1))


tDLS = DLS(M, y)

print(tDLS)

xeval = np.linspace(-np.pi, np.pi, 100)
yeval = evalTrigFunc(tDLS, xeval)

plt.plot(x,y, "o")
plt.plot(xeval,yeval)
plt.show()
