__author__ = 'Le'
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from math import sin
from math import pi
from math import e

def random_points(numbers):
    points = np.zeros((numbers, 4), dtype=np.float)
    points[:, 0:2] = np.random.uniform(-1.0, 1.0, (numbers, 2))
    points[:, 2] = [target_function(points[i, 0], points[i, 1]) for i in range(numbers)]
    return points


def target_function(x1, x2):
    if (x2 - x1 + 0.25 * sin(pi * x1)) < 0:
        return -1
    else:
        return 1


def clustering(k_clusters, points):
    """
    :param k_clusters:
    :param points:
    :return:k centroids
    """
    centroids = np.random.uniform(-1.0, 1.0, (k_clusters, 2))
    '''
    init k random centroids
    '''
    while True:
        points_in_cluster = np.zeros((k_clusters, 3), dtype=np.float)
        non_empty_clusters = 0
        is_cluster_change = False
        '''
        get k cluster
        '''
        for i in range(len(points)):
            min_dis = np.inf
            cluster = -1
            for k in range(k_clusters):
                dis_to_cluster_k = (points[i, 0] - centroids[k, 0])**2 + (points[i, 1] - centroids[k, 1])**2
                if dis_to_cluster_k < min_dis:
                    min_dis = dis_to_cluster_k
                    cluster = k
            if points_in_cluster[cluster, 2] == 0:
                non_empty_clusters += 1
            points_in_cluster[cluster, 2] += 1
            if points[i, 3] != cluster:
                is_cluster_change = True
            points[i, 3] = cluster
        if non_empty_clusters != k_clusters:
            return None
        if not is_cluster_change:
            return centroids
        '''
        recompute centroids
        '''
        for i in range(len(points)):
            points_in_cluster[points[i, 3], 0] += points[i, 0]
            points_in_cluster[points[i, 3], 1] += points[i, 1]
        for k in range(k_clusters):
            centroids[k, 0] = points_in_cluster[k, 0] / points_in_cluster[k, 2]
            centroids[k, 1] = points_in_cluster[k, 1] / points_in_cluster[k, 2]


def rbf_mat(centroids, points, gamma):
    k_centers = len(centroids)
    size = len(points)
    rbf_matrix = np.ones((size, k_centers + 1), dtype=np.float)
    for i in range(size):
        for k in range(k_centers):
            dis = (points[i, 0] - centroids[k, 0])**2 + (points[i, 1] - centroids[k, 1])**2
            rbf_matrix[i, k] = e**((-1) * gamma * dis)
    return rbf_matrix


def rbf_model(centroids, points, gamma):
    rbf_matrix = rbf_mat(centroids, points, gamma)
    lin_rgr = linear_model.LinearRegression(fit_intercept=False)
    lin_rgr.fit(rbf_matrix, points[:, 2])
    return lin_rgr


def error_rbf(hypothesis, test_points, centroids, gamma):
    size = len(test_points)
    mis_points = 0.0
    rbf_matrix = rbf_mat(centroids, test_points, gamma)
    for i in range(size):
        if hypothesis.predict(rbf_matrix[i, :]) * test_points[i, 2] < 0:
            mis_points += 1
    return mis_points / size


def svm_rbf_kernel(points, gamma):
    clf = svm.SVC(C=np.infty, kernel='rbf', gamma=gamma)
    clf.fit(points[:, 0:2], points[:, 2])
    if error_svm(clf, points) > 0.0:
        return None
    return clf


def error_svm(hypothesis, test_points):
    size = len(test_points)
    mis_points = 0.0
    for i in range(size):
        if hypothesis.predict(test_points[i, 0:2]) * test_points[i, 2] < 0:
            mis_points += 1
    return mis_points / size


def main():
    k = 9
    gamma = 1.5
    train_points = random_points(100)
    centers = clustering(k, train_points)
    while centers is None:
        centers = clustering(k, train_points)
    rbf_clf = rbf_model(centers, train_points, gamma)
    rbf_in_error = error_rbf(rbf_clf, train_points, centers, gamma)
    if rbf_in_error == 0:
        return 1
    else:
        return 0
count = 0
for i in range(1000):
    count += main()
print count