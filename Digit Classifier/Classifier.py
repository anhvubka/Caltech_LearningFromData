__author__ = 'Le'
import numpy as np
from sklearn import svm


def read_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    size = len(lines)
    training_data = np.zeros((size, 3), dtype=np.float)
    for i in range(size):
        line = lines[i]
        data = line.split()
        training_data[i, 0] = float(data[0])
        training_data[i, 1] = float(data[1])
        training_data[i, 2] = float(data[2])
    return training_data


def read_training():
    return read_file("features.train")


def read_test():
    return read_file("features.test")

train_data = read_training()
test_data = read_test()


def one_vs_all(label, C, kernel, degree, gamma, coef0, input_data):
    clf = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    data = np.ones((len(input_data), 1), dtype=np.float)
    for i in range(len(data)):
        if input_data[i, 0] != label:
            data[i, 0] = -1
    clf.fit(input_data[:, 1:3], data[:, 0])
    return clf


def one_vs_one(label1, label2, C, kernel, degree, gamma, coef0, data_source):
    clf = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    data = get_data_set(label1, label2, data_source)
    clf.fit(data[:, 1:3], data[:, 0])
    return clf


def error_one_vs_all(label, data_set, clf):
    data = np.ones((len(data_set), 1), dtype=np.float)
    mis_points = 0
    for i in range(len(data)):
        if clf.predict(data_set[i, 1:3]) == -1:
            if data_set[i, 0] == label:
                mis_points += 1
        else:
            if data_set[i, 0] != label:
                mis_points += 1
    return mis_points / float(len(data_set))


def error_one_vs_one(label1, label2, data_source, clf):
    mis_points = 0
    data_set = get_data_set(label1, label2, data_source)
    for i in range(len(data_set)):
        if data_set[i, 0] != clf.predict(data_set[i, 1:3]):
            mis_points += 1
    return mis_points / float(len(data_set))


def get_data_set(label1, label2, data_source):
    size = 0
    for i in range(len(data_source)):
        if data_source[i, 0] == label1 or data_source[i, 0] == label2:
            size += 1
    data = np.ones((size, 3), dtype=np.float)
    index = 0
    for i in range(len(data_source)):
        if data_source[i, 0] == label1:
            data[index, 0] = label1
            data[index, 1] = data_source[i, 1]
            data[index, 2] = data_source[i, 2]
            index += 1
        elif data_source[i, 0] == label2:
            data[index, 0] = label2
            data[index, 1] = data_source[i, 1]
            data[index, 2] = data_source[i, 2]
            index += 1
    return data


def get_train_data_set(label1, label2):
    return get_data_set(label1, label2, train_data)


def get_test_data_set(label1, label2):
    return get_data_set(label1, label2, test_data)


def get_cross_validation_index(index, size):
    cross_validation_size = size / 10
    remainder = size % 10
    if index < remainder:
        start_index = index * (cross_validation_size + 1)
        end_index = start_index + cross_validation_size + 1
    else:
        start_index = remainder * (cross_validation_size + 1) + (index - remainder) * cross_validation_size
        end_index = start_index + cross_validation_size
    return start_index, end_index


def get_validation_set(start, end, permutation_array, data_set):
    validation_size = end - start
    validation_set = np.ones((validation_size, 3), dtype=np.float)
    validation_index = 0
    train_set = np.ones((len(permutation_array) - validation_size, 3), dtype=np.float)
    train_index = 0
    for i in range(len(permutation_array)):
        if start <= i < end:
            validation_set[validation_index, :] = data_set[permutation_array[i], :]
            validation_index += 1
        else:
            train_set[train_index, :] = data_set[permutation_array[i], :]
            train_index += 1
    return validation_set, train_set
c_set = [0.01, 1.0, 100.0, 10000.0, 1000000.0]
for c in c_set:
    clf = one_vs_one(1.0, 5.0, c, 'rbf', 0.0, 1.0, 0.0, train_data)
    print error_one_vs_one(1.0, 5.0, train_data, clf), error_one_vs_one(1.0, 5.0, test_data, clf)
