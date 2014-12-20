__author__ = 'Le'
import numpy as np
import random as random
from sklearn import linear_model


def generate_points(n):
    points = np.ones((n, 4), dtype=np.float)
    points[:, 1:3] = np.random.uniform(-1, 1, (n, 2))
    for i in range(n):
        if (points[i, 1] ** 2 + points[i, 2] ** 2 - 0.6) > 0:
            points[i, 3] = 1
        else:
            points[i, 3] = -1
        n = random.randint(0, 9)
        if n == 0:
            points[i, 3] = -points[i, 3]

        #if points[i, 3] == 1:
            #plt.plot(points[i, 1], points[i, 2], 'ro')
        #else:
            #plt.plot(points[i, 1], points[i, 2], 'bo')
    return points


def linear_regression(points):
    linear_regr = linear_model.LinearRegression(fit_intercept=False)
    linear_regr.fit(points[:, 0:3], points[:, 3])
    print linear_regr.coef_
    return linear_regr.coef_


def get_misclassified_points(line, points):
    """

    :rtype : object
    """
    count = 0
    for i in range(len(points)):
        if (line[0] + line[1] * points[i, 1] + line[2] * points[i, 2]) > 0:
            if points[i, 3] == -1:
                count += 1
        else:
            if points[i, 3] == +1:
                count += 1
    return count


def get_error_rate(n):
    count = 0.0
    for i in range(n):
        points = generate_points(100)
        line = linear_regression(points)
        m = get_misclassified_points(line, points)
        count += m
        print m
    print count / n
    return count / n


def generate_nonlinear_points(n):
    points = np.ones((n, 7), dtype=np.float)
    points[:, 1:3] = np.random.uniform(-1, 1, (n, 2))
    for i in range(n):
        points[i, 3] = points[i, 1] * points[i, 2]
        points[i, 4] = points[i, 1] ** 2
        points[i, 5] = points[i, 2] ** 2
        if (points[i, 1] ** 2 + points[i, 2] ** 2 - 0.6) > 0:
            points[i, 6] = 1
        else:
            points[i, 6] = -1
        n = random.randint(0, 9)
        if n == 0:
            points[i, 6] = -points[i, 6]
    return points


def linear_regression_nonlinear_points(points):
    linear_regr = linear_model.LinearRegression(fit_intercept=False)
    linear_regr.fit(points[:, 0:6], points[:, 6])
    print linear_regr.coef_
    return linear_regr.coef_


def get_out_sample_error_rate(n, line):
    count = 0.0
    for i in range(n):
        points = generate_nonlinear_points(1000)
        m = nonlinear_misclassified_points(line, points)
        count += m
        print m
    print count / n
    return count / n


def nonlinear_misclassified_points(line, points):
    count = 0
    for i in range(len(points)):
        a = line.dot(points[i, 0:6])
        if a > 0:
            if points[i, 6] == -1:
                count += 1
        else:
            if points[i, 6] == 1:
                count += 1
    return count
points = generate_nonlinear_points(1000)
line0 = linear_regression_nonlinear_points(points)
get_out_sample_error_rate(1000, line0)

