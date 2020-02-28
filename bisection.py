from typing import Callable


def method(f: Callable, left=0.0, right=1.0, eps=0.001, *args) -> float:
    '''function that performs bisection method with given function'''
    if f(left, *args) * f(right, *args) > 0:
        return None
    else:
        mid = (right+left) / 2.0
        while (right-left) / 2.0 > eps:
            print(mid, eps)
            if abs(f(mid)) < eps:
                return mid
            elif f(left) * f(mid) < 0.0:
                '''there is a root between two points'''
                right = mid
            else:
                left = mid
            mid = (right+left) / 2.0
        return mid


if __name__ == "__main__":
    def f(x):
        return 5*x - 8
    print(method(f, -2.0, 2.0))
