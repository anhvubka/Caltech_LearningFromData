__author__ = 'Le'
import numpy as np
import random
from math import e
from math import sqrt
from math import log


class Line(object):
    def __init__(self, w0, w1, w2):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

    def is_above(self, x1, x2):
        if (self.w0 + self.w1 * x1 + self.w2 * x2) >= 0:
            return +1
        else:
            return -1


def random_line():
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    v0 = x2 * y1 - x1 * y2
    v1 = y2 - y1
    v2 = x1 - x2
    line = Line(v0, v1, v2)
    return line


def random_points(N, line):
    points = np.ones((N, 4), dtype=np.float)
    points[:, 1:3] = np.random.uniform(-1, 1, (N, 2))
    for i in range(N):
        points[i, 3] = line.is_above(points[i, 1], points[i, 2])
    return points


def stochastic_gradient_descent(n, line):
    points = random_points(n, line)
    w0 = [0.0, 0.0, 0.0]
    count = 0
    while True:
        perm = np.random.permutation(n)
        w1 = w0[:]
        for i in range(n):
            point = points[perm[i]]
            entropy = e**((-1) * point[3] * np.dot(w1, point[0:3]))
            for j in range(3):
                w1[j] -= 0.01 * (-1) * point[3] * point[j] * entropy / (1 + entropy)
        count += 1
        dist = 0.0
        for k in range(3):
            dist += (w1[k] - w0[k])**2
        dist = sqrt(dist)
        if dist >= 0.01:
            w0 = w1[:]
        else:
            break
    return w1


def get_out_sample_error(n):
    total = 0.0
    for j in range(n):
        line = random_line()
        points = random_points(1000, line)
        w = stochastic_gradient_descent(100, line)
        error = 0.0
        for i in range(1000):
            error += log(1 + e**((-1) * points[i, 3] * np.dot(w, points[i, 0:3])))
        error /= 1000
        total += error
        print error
    total = total / n
    print total
get_out_sample_error(100)