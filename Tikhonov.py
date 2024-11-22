import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import svd
from numpy.linalg import qr
from numpy.linalg import norm


def DLSQR(M, y):
    Q, R = qr(M)
    QT = Q.T   
    c = inv(R) @ QT @ y
    return c



def TikhonovRegression(M, y, l, G):
    MT = np.transpose(M)
    GT = np.transpose(G)
    c = inv(MT @ M + l * (GT @ G)) @ MT @ y
    return c

def TikhonovRegressionQR(M, y, l, G):
    Mp = np.block([[M],[np.sqrt(l)*G]])
    Z = np.zeros(G.shape[0])
    yp = np.block([y,Z])
    Q, R = qr(Mp)
    QT = Q.T   
    c = inv(R) @ QT @ yp
    return c

def derivativePolyPenaltyMatrix(x, m):
    D = np.zeros((x.size, m+1))
    for i in range(0, x.size):
        for j in range(1, m+1):
            D[i, j]  = j * x[i]**(j-1)
    return D
def derivativeTrigPenaltyMatrix(x, m):
    D = np.zeros((x.size, 2*m+1))
    for i in range(0, x.size):
        for k in range(1, m+1):
            D[i, 2*k -1]  = -k* np.sin(k * x[i]) 
            D[i, 2*k]  = k* np.cos(k * x[i])
    return D


def gramPenaltyMatrixPoly(x, m, e):
    M = GenMPolyBasis(x,m)
    U, s, VT = svd(M)
    L = s**2
    D = np.identity(m+1)
    l1 = L[0]
    for j in range(0, m+1):
        # D[j,j] = 1/L[j]
        if(L[j] > e * l1):
            D[j,j] = 0
        else:
            D[j,j] = 1
           
    return D @ VT    


def gramPenaltyMatrixTrig(x, m, e):
    M = GenMTrigBasis(x,m)
    # print(M)
    U, s, VT = svd(M)
    L = s**2
    print(s)
    D = np.identity(2*m+1)
    l1 = L[0]
    for j in range(0, 2*m+1):
        # D[j,j] = 1/L[j]
        if(L[j] > e * l1):
            D[j,j] = 0
        else:
            D[j,j] = 1
            
    print(D)
           
    return D @ VT   


def lowPassPenaltyMatrix(m, k0, c, d):
    G = np.zeros((2*m+1,2*m+1))
    for k in range(1, m+1):
        if k > k0:
            G[2*k -1, 2*k -1] = c*k**d
            G[2*k, 2*k]  = c*k**d 
    return G
    

def GenMPolyBasis(x, m):
    M = np.zeros((x.size, m+1))
    for i in range(0, x.size):
        for j in range(0, m+1):
            M[i, j]  = x[i]**j 
    return M



def GenMTrigBasis(x, m):
    M = np.ones((x.size, 2*m +1))
    for i in range(0, x.size):
        for k in range(1, m+1):
            M[i, 2*k -1]  = np.cos(k * x[i]) 
            M[i, 2*k]  = np.sin(k * x[i]) 
    return M



def evalTrigFunc(c, xeval):
    m = int((c.size -1)/2)
    y = np.zeros(xeval.size)
    
    for i in range(0, xeval.size):
        y[i] = c[0] 
        for k in range(1, m + 1):
                y[i]  += c[2*k -1]  * np.cos(k * xeval[i]) 
                y[i]  += c[2*k]  * np.sin(k * xeval[i]) 
    return y
                
def evalPolyFunc(c, xeval):
    return np.polyval(np.flip(c), xeval)
            

    
    