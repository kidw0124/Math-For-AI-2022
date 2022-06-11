#Gradient descent & Newton’s method with the backtracking linesearch

import sys


import sys
import numpy as np

def f(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def grad(x):
    return np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, -200*x[0]**2 + 200*x[1]])

def hess(x):
    return np.array([
        [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
        [-400*x[0], 200]
    ])
    
def norm(x):
    return (x[0]**2 + x[1]**2)**(1/2)

#Newton Direction
def pk(xk):
    return -(1/hess(xk))*grad(xk)
   
def linesearch(xk, f, grad, a0, p, c1, max_ls_iter):
    a = a0
    ls_iter = 0
    while ls_iter < max_ls_iter:
        #if(f(xk + a*pk(xk))<=f(xk) + c1*a*)
        pass
    
def minimization(x0, f, grad, hess, tol, max_iter):
    k = 0
    xk = x0
    print("x0 : " + str(x0) + ", ||grad(f(x0))||_2 : " + str(norm(grad(x0))) )
    while k < max_iter:
        #if norm(grad(x)) < tol:
            break
        
    return xk
        
    ``````