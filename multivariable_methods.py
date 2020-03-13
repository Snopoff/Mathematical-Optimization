import numpy as np
import autograd as ad
from typing import Callable


def Newton(f: Callable, x0: np.ndarray, eps=1.48e-08, maxiter=50) -> np.float:
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
        float : approximation for the root of the function

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


def main():
    def f(x):
        coefs = np.random.random(size=len(x))
        powers = np.random.randint(1, 5, size=len(x))
        X = np.array([x[i]**powers[i] for i in range(len(x))])
        return np.dot(X, coefs)

    x0 = np.random.random(5)

    print(Newton, x0)


if __name__ == "__main__":
    main()
