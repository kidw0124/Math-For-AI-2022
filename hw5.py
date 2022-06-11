#Gradient descent & Newton’s method with the backtracking linesearch

import sys
import numpy as np

def f(x):
    return np.array(100*(x[1]-x[0]**2)**2 + (1-x[0])**2)

def grad(x):
    return np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, -200*x[0]**2 + 200*x[1]])

def hess(x):
    return np.array([
        [1200*x[0][0]**2 - 400*x[1][0] + 2, -400*x[0][0]],
        [-400*x[0][0], 200]
    ])
    
def norm(x):
    #(x[0]**2 + x[1]**2)**(1/2)
    return np.linalg.norm(x,2)


def pk(xk, pk_type):
    #Newton Direction : pk_type = 0
    if not pk_type:
        return -grad(xk)/norm(grad(xk))
    #Steepest descent direction : pk_type = 1
    else:
        return -np.linalg.inv(hess(xk)).dot(grad(xk))
    

def linesearch(xk, a0, p, c1, max_ls_iter, pk_type):
    a = a0
    ls_iter = 0
    pk_ = pk(xk,pk_type)
    
    while ls_iter < max_ls_iter:
        if f(xk + a*pk_)<=f(xk) + c1*a*(grad(xk).T.dot(pk_)):
            break
        a = p*a
        ls_iter +=1
    return a

def minimization(x0, tol, max_iter, pk_type):
    k = 0
    xk = x0
    ak = 1.0
    while k < max_iter:
        if norm(grad(xk)) < tol:
            break
        print("Iteration %d: alp=%.4e, f=%.4e, gnorm=%.4e"%(k, ak, f(xk), norm(grad(xk))))
        #Choose a descent direction pk
        pk_ = pk(xk, pk_type)
        #Choos ak from direction
        ak = linesearch(xk, ak, p=0.5, c1=1e-4, max_ls_iter=max_iter/4, pk_type=pk_type)
        xk += ak*pk_
        k += 1
    return xk

def main():
    x1, x2 = map(float, input("input strating point (ex.1.2 1.2) : ").split(' '))
    print("\n")
    pk_type = int(input("choose a descent direction method (steepest descent : 0, newtern direction : 1) : "))
    print("\n")
    
    method = ""
    if not pk_type:
        method = "steepest descent"
    else:
        method = "newtern direction"
    
    x = np.array([[x1],[x2]])

    result = minimization(x, tol=1e-12,max_iter=50000, pk_type=pk_type)
    
    
    print("[Method] : "+ method + ", [Result] : (" + str(result[0][0]) + "," + str(result[1][0]) + ")" )

if __name__ == "__main__":
    main()
