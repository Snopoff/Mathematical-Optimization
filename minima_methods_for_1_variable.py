from typing import Callable
from math import sqrt

from numpy import linspace
from matplotlib.pyplot import plot, show


def bisection(f: Callable, left=0.0, right=1.0, eps=0.001) -> float:
    '''function that performs bisection method with given function'''
    if f(left) * f(right) > 0:
        return None
    else:
        mid = (right+left) / 2.0
        while (right-left) / 2.0 > eps:
            if abs(f(mid)) < eps:
                return mid
            elif f(left) * f(mid) < 0.0:
                '''there is a root between two points'''
                right = mid
            else:
                left = mid
            mid = (right+left) / 2.0
        return mid


def gss(f: Callable, left=0.0, right=1.0, eps=0.001) -> float:
    '''function that performs golden section search with given function'''
    gr = (1 + sqrt(5)) / 2
    c = right - (right - left) / gr
    d = left + (right - left) / gr
    while abs(c - d) > eps:
        if f(c) < f(d):
            left = d
        else:
            right = c

         # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = right - (right - left) / gr
        d = left + (right - left) / gr

    return (right + left) / 2


def plot_func(f: Callable, left=0.0, right=1.0):
    x = linspace(left, right)
    y = [f(a) for a in x]
    plot(x, y)
    show()


if __name__ == "__main__":
    def f(x):
        a = 5
        b = -2
        c = 3
        return a*x**2 + b*x + c
    print(bisection(f, -2.0, 2.0))
    print(gss(f, -2.0, 2.0))
