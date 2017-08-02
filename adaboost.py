# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    '''
    通过阈值比较对数据进行分类
    :return:
    '''
    return_array = ones((shape(data_matrix)[0],1))
    if thresh_ineq == 'lt':
        return_array[data_matrix[:, dimen] <= thresh_val] = 1.0
    else:
        return_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return return_array


def build_stump(data_array, class_labels, D):
    '''
    1.找到最佳单层决策树，由权重向量D决定
    :return:
    '''
    data_matrix = mat(data_array)
    label_mat = mat(class_labels).T
    m, n = shape(data_matrix)
    steps = 10.0
    best_stump = {}
    best_class_est = mat(zeros((m,1)))


if __name__ == '__main__':
    D = mat(ones((5,1))/5)
    data_array, class_labels = loadSimpData()













