from typing import Callable
from math import sqrt
import numpy as np
import autograd as ad


def bisection(f: Callable, a=0.0, b=1.0, eps=0.001) -> float:
    '''
    Bisection method

        Parameters
        ----------
        f : given function
        a : left border of an interval
        b : right border of an interval
        eps : accuracy

        Returns
        -------
        float : approximation for the root of a function

    '''
    if f(a) * f(b) > 0:
        return None
    else:
        mid = (b+a) / 2.0
        while (b-a) / 2.0 > eps:
            if abs(f(mid)) < eps:
                return mid
            elif f(a) * f(mid) < 0.0:
                '''there is a root between two points'''
                b = mid
            else:
                a = mid
            mid = (b+a) / 2.0
        return mid


def gss(f: Callable, a=0.0, b=1.0, eps=0.001) -> float:
    '''
    Golden section search

        Parameters
        ----------
        f : given function
        a : left border of an interval
        b : right border of an interval
        eps : accuracy

        Returns
        -------
        float : approximation for the minima of a function

    '''
    gr = (1 + sqrt(5)) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > eps:
        if f(c) < f(d):
            b = d
        else:
            a = c

         # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


def Simpson(f: Callable, a=0.0, b=1.0, n=100) -> float:
    '''
    Simpson's method

        Parameters
        ----------
        f : given function
        a : left border of an interval
        b : right border of an interval
        n : number of points 

        Returns
        -------
        float : approximation for the integral

    '''

    result = 1.0
    dx = (b - a)/n
    x = np.linspace(a, b, num=n)
    for i in range(n):
        result += f(x[i]) if i == 0 or i == n-1 else 2*((i % 2+1))*f(x[i])
    result *= dx/3
    return result


def Newton(f: Callable, x0=1.0, eps=1e-8, max_iter=1e6) -> float:
    '''
    Newton's method

        Parameters
        ----------
        f : given function
        x0 : initial guess for a solution f(x)=0
        eps : accuracy
        max_inter : maximum number of iterations

        Returns
        -------
        float : approximation for the root of the function

    '''

    df = ad.grad(f)

    result = x0
    for n in range(int(max_iter)):
        f_val = f(result)
        if abs(f_val) < eps:
            return result
        df_val = df(result)
        if df_val == 0:
            return None
        result = result - f_val/df_val

    return None


if __name__ == "__main__":
    def f(x):
        a = 5
        b = 1
        c = -3
        return a*x**2 + b*x + c

    a = 0.0
    b = 4.0
    print(bisection(f, a, b))
    print(gss(f, a, b))
    print(Simpson(f, a, b))
    print(Newton(f, df))
