import numpy as np
from numpy.linalg import svd 

def ridge_regression(M,y,l):
    MT = np.transpose(M)
    size=M.shape[1]
    U,D,V=svd(MT @ M + l*np.identity(size))
    D_inv=np.diag(1/D)
    c = V.T @ D_inv @ U.T @ MT @ y
    return c

def GenM(x, m):
    M = np.zeros((x.size, m+1))
    for i in range(0, x.size):
        for j in range(0, m+1):
            M[i, j]  = x[i]**j 
    return M

    
def evalmodel(c,x):
    yhat=np.zeros(len(x))
    for i in range(len(c)):
        yhat+=c[i]*x**i
    return yhat


def cv_lambda(l):
    return np.linspace(0,l,10*l+1)





    