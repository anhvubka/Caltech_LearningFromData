__author__ = 'Le'
import numpy as np
from sklearn import linear_model
from numpy.linalg import inv


def read_input_file(filename):
    f = open(filename, "r")
    lines = f.readlines()
    size = len(lines)
    output = np.zeros((size, 3), dtype=np.float)
    for i in range(size):
        line = lines[i]
        numbers = line.split()
        output[i, 0] = float(numbers[0])
        output[i, 1] = float(numbers[1])
        output[i, 2] = float(numbers[2])
    return output


def non_linear_transform(data):
    output = np.ones((len(data), 9), dtype=np.float)
    for i in range(len(data)):
        output[i, 1] = data[i, 0]
        output[i, 2] = data[i, 1]
        output[i, 3] = data[i, 0]**2
        output[i, 4] = data[i, 1]**2
        output[i, 5] = data[i, 0] * data[i, 1]
        output[i, 6] = np.abs(data[i, 0] - data[i, 1])
        output[i, 7] = np.abs(data[i, 0] + data[i, 1])
        output[i, 8] = data[i, 2]
    return output


def linear_regression(transformed_data):
    size = len(transformed_data)
    misclassified_points_in = 0.0
    out_sample_data = non_linear_transform(read_input_file("out.dta"))
    size_out = len(out_sample_data)
    misclassified_points_out = 0.0
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(transformed_data[:, 0:8], transformed_data[:, 8])
    for i in range(size):
        if (regr.predict(transformed_data[i, 0:8]) * transformed_data[i, 8]) < 0:
            misclassified_points_in += 1.0
    for j in range(size_out):
        if (regr.predict(out_sample_data[j, 0:8]) * out_sample_data[j, 8]) < 0:
            misclassified_points_out += 1.0
    return misclassified_points_in / size, misclassified_points_out / size_out


def linear_regression_validation(transformed_data, validation_size, model_k):
    training_size = len(transformed_data) - validation_size
    mis_validation_point = 0.0
    out_sample_data = non_linear_transform(read_input_file("out.dta"))
    size_out = len(out_sample_data)
    mis_out = 0.0
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(transformed_data[0:training_size, 0:(model_k + 1)], transformed_data[0:training_size, 8])
    for i in range(validation_size):
        if (regr.predict(transformed_data[training_size + i, 0:(model_k + 1)]) * transformed_data[training_size + i, 8]) < 0:
            mis_validation_point += 1.0
    for j in range(size_out):
        if (regr.predict(out_sample_data[j, 0:(model_k + 1)]) * out_sample_data[j, 8]) < 0:
            mis_out += 1.0
    print model_k, mis_validation_point / validation_size, mis_out / size_out


def reverse_training_set(transformed_data, validation_size, model_k):
    mis_validation_point = 0.0
    out_sample_data = non_linear_transform(read_input_file("out.dta"))
    size_out = len(out_sample_data)
    mis_out = 0.0
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(transformed_data[validation_size:, 0:(model_k + 1)], transformed_data[validation_size:, 8])
    for i in range(validation_size):
        if (regr.predict(transformed_data[i, 0:(model_k + 1)]) * transformed_data[i, 8]) < 0:
            mis_validation_point += 1.0
    for j in range(size_out):
        if (regr.predict(out_sample_data[j, 0:(model_k + 1)]) * out_sample_data[j, 8]) < 0:
            mis_out += 1.0
    print model_k, mis_validation_point / validation_size, mis_out / size_out

def regularization_regression(in_sample_data, out_sample_data, l):
    size_in = len(in_sample_data)
    size_out = len(out_sample_data)
    misclassified_points_in = 0.0
    misclassified_points_out = 0.0
    w_reg = np.dot(np.dot(inv(np.dot(in_sample_data[:, 0:8].transpose(), in_sample_data[:, 0:8]) + l * np.eye(8, dtype=np.float)), in_sample_data[:, 0:8].transpose()), in_sample_data[:, 8])
    for i in range(size_in):
        if (np.dot(w_reg, in_sample_data[i, 0:8]) * in_sample_data[i, 8]) < 0:
            misclassified_points_in += 1.0
    for j in range(size_out):
        if(np.dot(w_reg, out_sample_data[j, 0:8]) * out_sample_data[j, 8]) < 0:
            misclassified_points_out += 1.0
    print w_reg
    print misclassified_points_in / size_in, misclassified_points_out / size_out

data = non_linear_transform(read_input_file("in.dta"))
"""
for i in range(3, 8, 1):
    linear_regression_validation(data, 10, i)
print ''
for i in range(3, 8, 1):
    reverse_training_set(data, 25, i)
"""
total = 0.0
for i in range(10000):
    a = np.random.uniform(0.0, 1.0)
    b = np.random.uniform(0.0, 1.0)
    if a < b:
        total += a
    else:
        total += b
print total / 10000