
__author__ = 'Le'
import numpy as np
from sklearn import linear_model
from sklearn import svm


def read_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    size = len(lines)
    data = np.ones((size, 4), dtype=np.float)
    """
    0:label, 1:1, 2:x1, 3:x2
    """
    for i in range(size):
        line = lines[i]
        numbers = line.split()
        data[i, 0] = float(numbers[0])
        data[i, 2] = float(numbers[1])
        data[i, 3] = float(numbers[2])
    return data


def read_training():
    return read_file("features.train")


def read_test():
    return read_file("features.test")

training_set = read_training()
test_set = read_test()
svm_data = np.zeros((7, 3), dtype=np.float)
svm_data[0, 0] = 1.0
svm_data[0, 1] = 0.0
svm_data[0, 2] = -1.0

svm_data[1, 0] = 0.0
svm_data[1, 1] = 1.0
svm_data[1, 2] = -1.0

svm_data[2, 0] = 0.0
svm_data[2, 1] = -1.0
svm_data[2, 2] = -1.0

svm_data[3, 0] = -1.0
svm_data[3, 1] = 0.0
svm_data[3, 2] = 1.0

svm_data[4, 0] = 0.0
svm_data[4, 1] = 2.0
svm_data[4, 2] = 1.0

svm_data[5, 0] = 0.0
svm_data[5, 1] = -2.0
svm_data[5, 2] = 1.0

svm_data[6, 0] = -2.0
svm_data[6, 1] = 0.0
svm_data[6, 2] = 1.0


def get_one_vs_one_data(label1, label2, data_source):
    size = 0
    for i in range(len(data_source)):
        if data_source[i, 0] == label1 or data_source[i, 0] == label2:
            size += 1
    data = np.ones((size, 4), dtype=np.float)
    index = 0
    for i in range(len(data_source)):
        if data_source[i, 0] == label1:
            data[index, 0] = 1.0
            data[index, 2] = data_source[i, 2]
            data[index, 3] = data_source[i, 3]
            index += 1
        elif data_source[i, 0] == label2:
            data[index, 0] = -1.0
            data[index, 2] = data_source[i, 2]
            data[index, 3] = data_source[i, 3]
            index += 1
    return data


def get_one_vs_all_data(label, data_source):
    size = len(data_source)
    data = np.ones((size, 4), dtype=np.float)
    data[:, :] = data_source
    for i in range(size):
        if data_source[i, 0] == label:
            data[i, 0] = 1.0
        else:
            data[i, 0] = -1.0
    return data


def quadratic_transform(data_source):
    """
    :param data_source:
    0:label, 1:1, 2:x1, 3:x2
    :return:
    """
    size = len(data_source)
    transformed_data = np.ones((size, 7), dtype=np.float)
    """
    Transformed:
    0:label, 1:1, 2:x1, 3:x2, 4:x1x2, 5:x1**2, 6:x2**2
    """
    for i in range(size):
        transformed_data[i, 0] = data_source[i, 0]
        transformed_data[i, 2] = data_source[i, 2]
        transformed_data[i, 3] = data_source[i, 3]
        transformed_data[i, 4] = data_source[i, 2] * data_source[i, 3]
        transformed_data[i, 5] = data_source[i, 2]**2
        transformed_data[i, 6] = data_source[i, 3]**2
    return transformed_data


def fit_data(data_source, reg_para):
    clf = linear_model.Ridge(alpha=reg_para, fit_intercept=False)
    clf.fit(data_source[:, 1:], data_source[:, 0])
    return clf


def get_error(clf, data_source):
    mis_points = 0.0
    size = len(data_source)
    for i in range(size):
        if clf.predict(data_source[i, 1:]) * data_source[i, 0] < 0:
            mis_points += 1
    print mis_points,

clf = svm.SVC(C=np.inf, kernel='poly', degree=2.0, gamma=1.0, coef0=1.0)
clf.fit(svm_data[:, 0:2], svm_data[:, 2])
print clf.dual_coef_
