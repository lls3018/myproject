# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('C:\Users\user\Desktop\work\projects\myproject\data\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inx):
    return 1.0/(1+exp(-inx))


def classify_vector(inx, weights):
    prob = sigmoid(sum(inx*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def grad_ascent(data_mat_in, class_labels):
    '''
    梯度上升算法
    '''
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    #print "label_mat",label_mat
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    #print weights
    for k in range(max_cycles):
        h = sigmoid(data_matrix*weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stochastic_grad_ascent0(data_matrix, class_labels):
    '''
    随机梯度上升算法
    '''
    # 计算矩阵的m行数，和n列数
    m, n = shape(data_matrix)
    print "data_matrix",data_matrix
    alpha = 0.01
    weight = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i]*weight))
        error = class_labels[i] - h
        weight = weight + alpha * error * data_matrix[i]
    return weight


def stochastic_grad_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            # 每次迭代时需要调整
            alpha = 4/(1.0 + j + i) + 0.4
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index]*weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del data_index[rand_index]
    return weights


def colic_test():
    f_train = open('C:\Users\user\Desktop\work\projects\myproject\data\horseColicTraining.txt')
    f_test = open('C:\Users\user\Desktop\work\projects\myproject\data\horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in f_train.readlines():
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(current_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(current_line[21]))
    #print array(training_set)
    trans_weights = stochastic_grad_ascent0(array(training_set), training_labels)
    print trans_weights


def plot_best_fit(weights):
    data_mat, label_mat = loadDataSet()
    data_array = array(data_mat)
    n = shape(data_array)[0]
    x_cord1, y_cord1, x_cord2, y_cord2 = [], [], [], []
    for i in range(n):
        if int(label_mat[i] == 1):
            x_cord1.append(data_array[i, 1])
            y_cord1.append(data_array[i, 2])
        else:
            x_cord2.append(data_array[i, 1])
            y_cord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = arange(-3.0, 3.0, -0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    data_array, label_mat = loadDataSet()
    weights = stochastic_grad_ascent1(array(data_array), label_mat)
    plot_best_fit(weights)


