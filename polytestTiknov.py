from Tikhonov import *
import numpy as np

np.random.seed(13)
# f = lambda x: 0.000383962* x**5-0.019501*x**4+0.345199*x**3-2.4914*x**2+6.67878*x-6.78197
# f = lambda x: 0.17473* x**5-1.83319*x**4+6.13346*x**3-6.36251*x**2+0.147673
# f = lambda x: 0.13632 * x**5-0.915182*x**4+8.93092*x**2-13.41581*x+3.1683
f = lambda x: 0.217979 * x**5-2.34262*x**4+8.16723*x**3-9.33027*x**2+1.79156
# f = lambda x: 12.05439 * x**5-34.26494*x**3+ 26.93923*x**2-4.50066*x-0.177413

a = 0
b = 4.7
x = np.linspace(a, b, 200)
y =  f(x)+np.random.randn(200)



xVal = x[::2]
yVal = y[::2]

xTrain = x[1::2]
yTrain = y[1::2]
m = 21
l = np.arange(0, 0.2, 0.0001)
sqErr = np.zeros(l.size)

M = GenMPolyBasis(xTrain, m)
# G = gramPenaltyMatrix(xTrain, m, 0.001)
G = derivativePolyPenaltyMatrix(xTrain, m)

for i in range(0, l.size):
    pTik = TikhonovRegressionQR(M, yTrain, l[i], G)
    yeval = evalPolyFunc(pTik, xVal);
    sqErr[i] = norm(yVal - yeval)**2
plt.semilogy(l,sqErr, "o")
plt.xlabel("ln(λ)")
plt.ylabel("ln(Sqr Err)")
plt.show()

# for i in range(0, l.size):
#     pTik = TikhonovRegressionQR(M, yTrain, l[i], G)
#     yeval = evalPolyFunc(pTik, xVal);
#     sqErr[i] = norm(yVal - yeval)**2
# plt.semilogy(l,sqErr, "o")
# plt.show()

pTik2 = TikhonovRegressionQR(M, yTrain, l[np.argmin(sqErr)], G)
# pTik2 = TikhonovRegression(M, yTrain, 0.4, G)
# pTik1 = TikhonovRegression(M, yTrain, 0.4, G)
# pTik2 = TikhonovRegressionQR(M, yTrain, 0.4, G)

# pTik1 = DLSSVD(M, yTrain)
pTik1 = DLSQR(M, yTrain)



# p2 = TikhonovRegression(GenMPolyBasis(x, m+1), y)
# p2 = DLS(GenMPolyBasis(x))

xeval =  np.linspace(a, b , 1000);

yevalTik1  = evalPolyFunc(pTik1, xeval)
yevalTik2  = evalPolyFunc(pTik2, xeval)


plt.plot(xeval,f(xeval), label = "f(x)")
plt.plot(xVal,yVal, 'o', label = "Validation")
plt.plot(xTrain,yTrain, 'x', label = "Training")
# plt.plot(xeval,yevalDLS)
plt.plot(xeval,yevalTik2, label = f"λ = {l[np.argmin(sqErr)]}")
# plt.plot(xeval,yevalTik1, label = "SVD")
plt.plot(xeval,yevalTik1, label = "QR")


plt.legend()
plt.show()

yerreval1 = evalPolyFunc(pTik1, xVal);
yerreval2 = evalPolyFunc(pTik2, xVal);

print(f"DLS error =  {norm(yVal - yerreval1)**2}")
print(f"Tikhonov err =  {norm(yVal - yerreval2)**2}")