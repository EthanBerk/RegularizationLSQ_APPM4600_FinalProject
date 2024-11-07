import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.polynomial import *

def main() :
    x = np.array([1,2,3])
    y = np.array([2,4,5])

    p1 = DLS(x, y, 4)
    
    
    xeval, yeval  = p1.linspace(1000, [0,3])

    print(p1)
    
    plt.plot(x,y, "o")
    plt.plot(xeval,yeval)
    plt.show()
    
    
    
    
    

def DLS(x, y, n):
    M = GenM(x,n)
    MT = np.transpose(M)
    c = inv(MT @ M) @ MT @ y
    return Polynomial(c)

def RidgeRegression(x, y, l, n):
    M = GenM(x,n)
    I = np.identity(n)
    MT = np.transpose(M)
    c = inv(MT @ M - l * I) @ MT @ y
    return Polynomial(c)
    
    
def GenM(x, n):
    M = np.zeros((x.size, n))
    for i in range(0, x.size):
        for j in range(0, n):
            M[i, j]  = x[i]**j 
    return M

main()

    
    