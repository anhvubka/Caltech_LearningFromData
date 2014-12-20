__author__ = 'Le'
from math import log
from math import sin
from math import sqrt
from math import pi
import random


def vc_bound(dvc, N, delta):
    return sqrt(8/N * (log(4 / delta) + dvc * log(2 * N)))


def rademacher_bound(dvc, N, delta):
    return sqrt(2 / N * (log(2 * N) + dvc * log(N))) + sqrt((2 / N) * log(1/delta)) + 1 / N


def parondo_bound(dvc, N, delta):
    return (sqrt(1 + N * (log(6 / delta) + dvc * log(2 * N))) + 1) / N


def devroye_bound(dvc, N, delta):
    return (sqrt(1 + (log(4 / delta) + 2 * dvc * log(N)) * (N - 2) / 2) + 1) / (N - 2)


def get_ax(x1, x2):
    return (x1 * sin(pi * x1) + x2 * sin(pi * x2)) / (x1**2 + x2**2)


def get_square_error(a):
    return (2 * a**2 / 3 + 1 - 4 * a / pi) / 2

def get_variance(a1, a):
    return (a1 - a)**2 /3
total = 0
for i in range(10000):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    total += get_ax(x1, x2)
a = total / 10000.0
total = 0
print a
for i in range(10000):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    a1 = get_ax(x1, x2)
    total += get_variance(a1,a)
print total/10000.0