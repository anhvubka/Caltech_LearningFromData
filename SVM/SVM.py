__author__ = 'Le'
import random
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

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


def get_misclassified(l, points):
    mis_points = []
    count = 0
    for j in range(len(points)):
        if l.is_above(points[j, 1], points[j, 2]) != points[j, 3]:
            count += 1
            mis_points.append(points[j])
    return count, mis_points


def get_mis_points(l, points):
    count = 0
    for j in range(len(points)):
        if l.is_above(points[j, 1], points[j, 2]) != points[j, 3]:
            count += 1
    return count


def random_points(N, line):
    is_dicard = True
    points = np.zeros((N, 4), dtype=np.float)
    while is_dicard:
        points = np.zeros((N, 4), dtype=np.float)
        points[:, 1:3] = np.random.uniform(-1, 1, (N, 2))
        for i in range(N):
            points[i, 3] = line.is_above(points[i, 1], points[i, 2])
            if i > 0:
                if points[i, 3] != points[0, 3]:
                    is_dicard = False
            points[i, 0] = 1
    return points


def compare_pla_svm(n_in, n_out):
    line = random_line()
    points = random_points(n_in, line)
    line0 = Line(0, 0, 0)
    mis_points = get_misclassified(line0, points)[1]
    while len(mis_points) > 0:
        if len(mis_points) == 1:
            p = 0
        else:
            p = random.randrange(0, len(mis_points) - 1)
        line0.w0 = line0.w0 + mis_points[p][0] * mis_points[p][3]
        line0.w1 = line0.w1 + mis_points[p][1] * mis_points[p][3]
        line0.w2 = line0.w2 + mis_points[p][2] * mis_points[p][3]
        mis_points = get_misclassified(line0, points)[1]
    test_points = random_points(n_out, line)
    m = get_mis_points(line0, test_points)
    alpha = quad_programming(points)
    w_1 = 0
    w_2 = 0
    w_0 = 0
    for i in range(len(alpha)):
        if abs(alpha[i]) < 1.0e-04:
            alpha[i] = 0
        else:
            w_1 += alpha[i] * points[i, 1] * points[i, 3]
            w_2 += alpha[i] * points[i, 2] * points[i, 3]
    for j in range(len(alpha)):
        if abs(alpha[j]) > 1.0:
            w_0 = (1.0 / points[j, 3]) - (w_1 * points[j, 1] + w_2 * points[j, 2])
            break
    line1 = Line(w_0, w_1, w_2)
    m1 = get_mis_points(line1, test_points)
    print m1
    if m1 < m:
        return 1
    else:
        return 0


def quad_programming(points):
    N = len(points)
    b = matrix(np.zeros((1, 1), dtype=np.float), tc='d')
    h = matrix(np.zeros(N))
    q = matrix(-np.ones((N, 1), dtype=np.float), tc='d')
    a1 = np.array([points[:, 3]])
    A = matrix(a1, tc='d')
    G = matrix(np.diag(np.ones(N) * -1))
    p1 = np.zeros((N, N), dtype=np.float)
    for i in range(N):
        for j in range(N):
            p1[i, j] = points[i, 3] * points[j, 3] * (points[i, 1] * points[j, 1] + points[i, 2] * points[j, 2])
    P = matrix(p1, tc='d')
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    return sol['x']

total = 0
for i in range(1000):
    total += compare_pla_svm(100, 10000)
print total
