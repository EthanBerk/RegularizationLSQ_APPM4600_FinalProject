from Tikhonov import *
import numpy as np

np.random.seed(10)
f = lambda x: 0.0000997 * x**5-0.010879*x**4+0.4439*x**3-8.31*x**2+69.75*x-192
a = 5
b = 38
x = np.linspace(a, b, 200)
y =  f(x)+np.random.randn(200)



xVal = x[::2]
yVal = y[::2]

xTrain = x[1::2]
yTrain = y[1::2]

m = 1
l = np.arange(0, 4, 0.1)
sqErr = np.zeros(l.size)

M = GenMPolyBasis(xTrain, m+1)
G = gramPenaltyMatrix(xTrain, m+1, 0.4)

for i in range(0, l.size):
    pTik = TikhonovRegression(M, yTrain, l[i], G)
    yeval = evalPolyFunc(pTik, xVal);
    sqErr[i] = norm(yVal - yeval)**2
plt.semilogy(l,sqErr, "o")
plt.show()

pTik1 = TikhonovRegression(M, yTrain, 0, G)
pTik2 = TikhonovRegression(M, yTrain, 3, G)


# p2 = TikhonovRegression(GenMPolyBasis(x, m+1), y)
# p2 = DLS(GenMPolyBasis(x))

xeval =  np.linspace(5, 38 , 1000);

yevalTik1  = evalPolyFunc(pTik1, xeval)
yevalTik2  = evalPolyFunc(pTik2, xeval)


plt.plot(xVal,yVal, "o")
plt.plot(xTrain,yTrain, "o")
# plt.plot(xeval,yevalDLS)
plt.plot(xeval,yevalTik1)
plt.plot(xeval,yevalTik2)
plt.show()