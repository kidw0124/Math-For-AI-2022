
import sys
import numpy as np


def norm(x):
    #(x[0]**2 + x[1]**2)**(1/2)
    return np.linalg.norm(x, 2)


def backtracking_Linesearch(x_k, p_k, f, grad, alpha_0=1.0, rho=0.5, c_1=1e-4, max_iter=12500):
    alpha = alpha_0
    iter = 0
    while iter < max_iter:
        if f(x_k + alpha*p_k) <= f(x_k) + c_1*alpha*(grad(x_k).T).dot(p_k):
            break
        alpha = rho*alpha
        iter += 1
    return alpha


def do(type, x_0, f, grad, hess, tol=1e-12, max_iter=50000, file=None):
    k = 0
    x_k = x_0
    while k < max_iter:
        if norm(grad(x_k)) < tol:
            break
        # Choose a descent direction p_k
        if type == 'steepest':
            p_k = -grad(x_k)/norm(grad(x_k))
        elif type == 'newton':
            p_k = -np.linalg.inv(hess(x_k)).dot(grad(x_k))
        # Choose a step size alpha_k
        alpha_k = backtracking_Linesearch(x_k, p_k, f, grad)
        print("Iteration: ", k, "alpha: ", alpha_k,
              "f(x_k): ", f(x_k), "gnorm: ", norm(grad(x_k)), file=file)
        # Update x_k
        x_k = x_k + alpha_k*p_k
        k += 1
    return x_k


def steepest_descent(x_0, f, grad, hess, tol=1e-12, max_iter=50000, file=None):
    return do('steepest', x_0, f, grad, hess, tol, max_iter, file)


def newton_method(x_0, f, grad, hess, tol=1e-12, max_iter=50000, file=None):
    return do('newton', x_0, f, grad, hess, tol, max_iter, file)


def Rosenbrock(x):
    return (1 - x[0][0])**2 + 100*(x[1][0] - x[0][0]**2)**2


def Gradient_Rosenbrock(x):
    return np.array([[400*x[0][0]**3-400*x[0][0]*x[1][0]+2*x[0][0]-2], [200*x[1][0]-200*x[0][0]**2]])


def Hessian_Rosenbrock(x):
    return np.array([[1200*x[0][0]**2-400*x[1][0]+2, -400*x[0][0]], [-400*x[0][0], 200]])

def __main__():
    path ="output"
    for x_0, name in [[np.array([[1.2], [1.2]]),"S0"], [np.array([[-1.2],[1]]),"S1"]]:
        f = open(path + "/GD_" + name + ".txt", "w")
        print("x_0^T: ", x_0.T, file=f)
        x_k = steepest_descent(x_0, Rosenbrock, Gradient_Rosenbrock, Hessian_Rosenbrock, file=f)
        print("Steepest Descent: ", x_k.T, "^T", file=f)
        f.close()
        f= open(path + "/NT_" + name + ".txt", "a")
        print("x_0^T: ", x_0.T, file=f)
        x_k = newton_method(x_0, Rosenbrock, Gradient_Rosenbrock, Hessian_Rosenbrock, file=f)
        print("Newton Method: ", x_k.T, "^T", file=f)
        f.close()

__main__()
