import numpy as np
import autograd as ad
from typing import Callable


def Newton(f: Callable, x0: np.ndarray, eps=1.48e-08, maxiter=50) -> np.array:
    '''
    Newton's method

        Parameters
        ----------
        f : given function
        x0 : initial guess for a solution f(x)=0
        eps : accuracy
        max_iter : maximum number of iterations

        Returns
        -------
        np.array : approximation for the minimum of the (compact) function

    '''
    g = ad.grad(f)
    h = ad.hessian(f)

    x = x0
    for _ in range(maxiter):
        delta = np.linalg.solve(h(x), -g(x))
        x = x + delta
        if np.linalg.norm(delta) < eps:
            break

    return x


def descent(f: Callable, x0: np.array, eps=10e-5, lr=10e-3, maxiter=100) -> np.array:
    """
    Gradient Descent

        Parameters
        ----------
        f : cost function to minimize
        x0 : initial point
        eps : accuracy
        lr : descent(learning) rate
        maxiter : max number of iterations

        Returns
        -------
        np.array : approximation for the minimum of the function on compact(convex) area
    """
    x = x0
    df = ad.grad(f)  # / np.linalg.norm(ad.grad(f))
    for _ in range(maxiter):
        curr_x = x
        x = curr_x - np.multiply(lr, df(curr_x))

        try:
            if np.linalg.norm(x - curr_x) <= eps:
                return x
        except RuntimeWarning:
            return np.array([np.inf]*len(x))

    return x


def Penalty(f: Callable, c: np.array, x0: np.array) -> np.array:
    """
    Penalty method

        Parameters
        ----------
        f : cost function to minimize
        c : set of constraint functions
        x0 : initial point

        Returns
        -------
        np.array : approximation for the minimum of the function on compact(convex) area

    """
    def J(x, k=1): return f(x) + sum([k*func(x) for func in c])
    #dJ = lambda x: ad.grad(f)(x) + sum([ad.grad(func)(x) for func in c])
    return descent(J, x0)


def main():
    def f(x):
        coefs = np.random.random(size=len(x))
        powers = np.random.randint(1, 5, size=len(x))
        X = np.array([x[i]**powers[i] for i in range(len(x))])
        # return np.dot(X, coefs)
        return (x[0] ** 3) + (x[1] ** 3) - (9 * x[0] * x[1]) + 27

    x0 = np.random.random(2)

    def constr_1(x):
        return 2*x[0] + 4*x[1] - 5

    def constr_2(x):
        return 9*x[0] + x[1] - 10

    print(Newton(f, x0))
    print(Penalty(f, [constr_1, constr_2], [-1.0, 1.0]))


if __name__ == "__main__":
    main()
