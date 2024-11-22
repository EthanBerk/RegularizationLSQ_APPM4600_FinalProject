from Ridge import *
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#part A and B
np.random.seed(1)
f = lambda x: x**2
a = 0
b = 5
x = np.linspace(a, b, 200)
y =  f(x)+np.random.randn(200)

df=np.column_stack((x,y))

train, test = train_test_split(df, test_size=0.5)

#part  C
M=GenM(train[:,0],5)

beta=ridge_regression(M,train[:,1],0)

y_hat=evalmodel(beta,test[:,0])

print('SSE =',sum((test[:,1]-y_hat)**2))

f=evalmodel(beta,x)

plt.plot(train[:,0],train[:,1],linestyle='None',marker='x',ms=5,color='cyan',label='Training')
plt.plot(test[:,0],test[:,1],linestyle='None',marker='o',ms=5,color='orange',label='Testing')
plt.plot(x,f,linestyle='-',label='Model')
plt.title('Ordinary Least Squares Model')
plt.legend()
plt.show()

lam=cv_lambda(10)
SSE=[]
for i in lam:

    beta=ridge_regression(M,train[:,1],i)

    y_hat=evalmodel(beta,test[:,0])
    SSE.append(sum((test[:,1]-y_hat)**2))
    
plt.plot(lam,SSE)
plt.title('Calculating Best lambda')
plt.xlabel('Lambda')
plt.ylabel('Sum of Squared Error')
plt.show()
print('min SSE = ',SSE[np.argmin(SSE)],'lambda = ',lam[np.argmin(SSE)])

# Part E

beta_ols=ridge_regression(M,train[:,1],0)
beta_best=ridge_regression(M,train[:,1],lam[np.argmin(SSE)])
beta_ridge=ridge_regression(M,train[:,1],100)

y_hat_ols=evalmodel(beta_ols,test[:,0])
y_hat_best=evalmodel(beta_best,test[:,0])
y_hat_ridge=evalmodel(beta_ridge,test[:,0])

f_ols=evalmodel(beta_ols,x)
f_best=evalmodel(beta_best,x)
f_ridge=evalmodel(beta_ridge,x)

plt.plot(train[:,0],train[:,1],linestyle='None',marker='x',ms=5,color='cyan',label='Training')
plt.plot(test[:,0],test[:,1],linestyle='None',marker='o',ms=5,color='orange',label='Testing')
plt.plot(x,f_ols,linestyle='-',label='OLS Model',linewidth=3)
plt.plot(x,f_best,linestyle='-',color='red',label='Best Model')
plt.plot(x,f_ridge,linestyle='-',color='green',label='Large Lambda Model')
plt.title('Ridge Model with lambda=2.2')
plt.legend()

