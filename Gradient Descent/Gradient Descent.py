__author__ = 'Le'
from math import e
import numpy as np


def error_function(u, v):
    return (u * (e**v) - 2 * v / (e**u))**2


def partial_derivative_u(u, v):
    return 2 * (e**v + 2 * v / (e**u)) * (u * (e**v) - 2 * v / (e**u))


def partial_derivative_v(u, v):
    return 2 * (u * (e**v) - 2 * v / (e**u)) * (u * (e**v) - 2 / (e**u))


def gradient_descent(u, v, error_threshold):
    count = 0
    while error_function(u, v) >= error_threshold:
        gradient_u = partial_derivative_u(u, v)
        gradient_v = partial_derivative_v(u, v)
        u -= 0.1 * gradient_u
        v -= 0.1 * gradient_v
        count += 1
        print u, v, error_function(u, v)
    return u, v, count


def coordinate_descent(u, v, iteration):
    count = 0
    tmp_u = u
    tmp_v = v
    while count < iteration:
        gradient_u = partial_derivative_u(tmp_u, tmp_v)
        tmp_u -= 0.1 * gradient_u
        gradient_v = partial_derivative_v(tmp_u, tmp_v)
        tmp_v -= 0.1 * gradient_v
        print tmp_u, tmp_v
        count += 1
        print error_function(tmp_u, tmp_v)
    return error_function(tmp_u, tmp_v)

x = np.float64(1.0)
m = np.float64(10**(-14))
coordinate_descent(x, x, 15)


