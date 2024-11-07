import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 

def main() :
    x = np.array([1,2,3])
    y = np.array([2,4,5])

    print(GenM(x, 2))
    print(DLS(x, y, 2))
    print(RidgeRegression(x, y, 2, 2))
    
    
    

def DLS(x, y, n):
    M = GenM(x,n)
    MT = np.transpose(M)
    return inv(MT @ M) @ MT @ y

def RidgeRegression(x, y, l, n):
    M = GenM(x,n)
    I = np.identity(n)
    MT = np.transpose(M)
    return inv(MT @ M - l * I) @ MT @ y
    
    
def GenM(x, n):
    M = np.zeros((x.size, n))
    for i in range(0, x.size):
        for j in range(0, n):
            M[i, j]  = x[i]**j 
    return M

main()

    
    