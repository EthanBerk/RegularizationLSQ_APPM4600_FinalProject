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
x = np.linspace(a, b, 100)
y =  f(x)+np.random.randn(100)



xVal = x[::2]
yVal = y[::2]

xTrain = x[1::2]
yTrain = y[1::2]


xeval = np.linspace(a,b,1000)
plt.plot(xeval, f(xeval), label = "f(x)")


plt.plot(xVal,yVal, 'o', label = "Validation")
plt.plot(xTrain,yTrain, 'x', label = "Training")



plt.legend()
plt.show()

yerreval1 = evalPolyFunc(pTik1, xVal);
yerreval2 = evalPolyFunc(pTik2, xVal);

print(f"DLS error =  {norm(yVal - yerreval1)**2}")
print(f"Tikhonov err =  {norm(yVal - yerreval2)**2}")