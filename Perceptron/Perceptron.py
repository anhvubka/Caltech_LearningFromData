__author__ = 'Le'
import random
import numpy as np
from sklearn import linear_model


class Line(object):
    def __init__(self, w0, w1, w2):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

    def is_above(self, x, y):
        if (self.w0 + self.w1 * x + self.w2 * y) >= 0:
            return +1
        else:
            return -1


class Point(object):
    is_above = -1

    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_misclassified(l, points):
    mis_points = []
    count = 0
    for j in range(len(points)):
        if l.is_above(points[j, 1], points[j, 2]) != points[j, 3]:
            count += 1
            mis_points.append(points[j])
    return count, mis_points


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
    points = np.zeros((N, 4), dtype=np.float)
    points[:, 1:3] = np.random.uniform(-1, 1, (N, 2))
    red_points = 0
    blue_points = 0
    for i in range(N):
        points[i, 3] = line.is_above(points[i, 1], points[i, 2])
        points[i, 0] = 1
        if points[i, 3] == 1:
            red_points += 1
        else:
            blue_points += 1
    return points


def get_lr_line(points):
    linear_regr = linear_model.LinearRegression(fit_intercept=False)
    linear_regr.fit(points[:, 0:3], points[:, 3])
    line0 = Line(linear_regr.coef_[0], linear_regr.coef_[1], linear_regr.coef_[2])
    return line0


def test_in_sample_error(n_in):
    line = random_line()
    points = random_points(n_in, line)
    line0 = get_lr_line(points)
    m = get_misclassified(line0, points)[0]
    print m
    return m


def test_out_sample_error(n_in, n_out):
    line = random_line()
    points = random_points(n_in, line)
    line0 = get_lr_line(points)
    test_points = random_points(n_out, line)
    m = get_misclassified(line0, test_points)[0]
    print m
    return m


def perceptron_lr(n_in):
    line = random_line()
    points = random_points(n_in, line)
    line0 = get_lr_line(points)
    m = get_misclassified(line0, points)[0]
    mis_points = get_misclassified(line0, points)[1]
    count = 0
    while len(mis_points) > 0:
        if len(mis_points) == 1:
            p = 0
        else:
            p = random.randrange(0, len(mis_points) - 1)
        line0.w0 = line0.w0 + mis_points[p][0] * mis_points[p][3]
        line0.w1 = line0.w1 + mis_points[p][1] * mis_points[p][3]
        line0.w2 = line0.w2 + mis_points[p][2] * mis_points[p][3]
        mis_points = get_misclassified(line0, points)[1]
        count += 1
    print count
    return count

count = 0
for i in range(1000):
    count += perceptron_lr(10)
print count


