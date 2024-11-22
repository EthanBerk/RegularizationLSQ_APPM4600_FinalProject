from Tikhonov import *
import numpy as np
np.random.seed(15)
f = lambda x: 2*(1  -0.53 * np.sin(3*x) +0.2*np.cos(3*x)-0.32 * np.sin(6*x) +0.53 *np.cos(6*x))
a = -np.pi
b = np.pi
x = np.linspace(a, b, 200)
y =  f(x)+ 4*np.random.randn(200)



xVal = x[::2]
yVal = y[::2]

xTrain = x[1::2]
yTrain = y[1::2]

m = 20
l = np.arange(6, 8, 0.01)
# l = np.arange(0, 0.01, 0.0001)
sqErr = np.zeros(l.size)

M = GenMTrigBasis(xTrain, m)
G = lowPassPenaltyMatrix(m, 6, 1, 1)
# G = derivativeTrigPenaltyMatrix(x,m )

for i in range(0, l.size):
    pTik = TikhonovRegressionQR(M, yTrain, l[i], G)
    yeval = evalTrigFunc(pTik, xVal);
    sqErr[i] = norm(yVal - yeval)**2

plt.semilogy(l,sqErr, "o")
plt.xlabel("λ")
plt.ylabel("ln(Sqr Err)")
plt.show()


pTik2 = TikhonovRegressionQR(M, yTrain, l[np.argmin(sqErr)], G)
# pTik2 = TikhonovRegression(M, yTrain, 0.4, G)
# pTik1 = TikhonovRegression(M, yTrain, 0.4, G)
# pTik2 = TikhonovRegressionQR(M, yTrain, 0.4, G)
# 6.989999999999979

# pTik1 = DLSSVD(M, yTrain)
pTik1 = DLSQR(M, yTrain)



# p2 = TikhonovRegression(GenMPolyBasis(x, m+1), y)
# p2 = DLS(GenMPolyBasis(x))

xeval =  np.linspace(a, b , 1000);

yevalTik1  = evalTrigFunc(pTik1, xeval)
yevalTik2  = evalTrigFunc(pTik2, xeval)


plt.plot(xeval,f(xeval), label = "f(x)")
# plt.plot(xeval,f(xeval), label = "f(x)")
# plt.plot(xVal,yVal, 'o', label = "Validation")
# plt.plot(xTrain,yTrain, 'x', label = "Training")
# # plt.plot(xeval,yevalDLS)
plt.plot(xeval,yevalTik2, label = f"Tikhnov: λ = {l[np.argmin(sqErr)]}")
# # plt.plot(xeval,yevalTik1, label = "SVD")
plt.plot(xeval,yevalTik1, label = "QR")


plt.legend()
plt.show()

yerreval1 = evalPolyFunc(pTik1, xVal);
yerreval2 = evalPolyFunc(pTik2, xVal);

print(f"DLS error =  {norm(f(xeval) - yevalTik1)**2}")
print(f"Tikhonov err =  {norm(f(xeval) - yevalTik2)**2}")
print(f"Tikhonov l =  {l[np.argmin(sqErr)]}")
print({l[np.argmin(sqErr)]})