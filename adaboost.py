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
    将最小错误率min_error 设置为无穷大
    对数据集中的每一个特征（第一层循环）
        对每个步长（第二层循环）
            对每个不等号（第三层循环）
                建立一个单层决策树并利用加权对数据集进行测试
                如果错误率低于min_error, 则将当前单层决策树设置为最佳单层决策树
    返回最佳单层决策树
    '''
    data_matrix = mat(data_array)
    label_mat = mat(class_labels).T
    m, n = shape(data_matrix)
    steps = 10.0
    best_stump = {}
    best_class_est = mat(zeros((m,1)))
    min_error = inf
    for i in range(n):

        range_min = data_matrix[:,i].min()
        range_max = data_matrix[:,i].max()
        step_size = (range_max - range_min)/steps
        for j in range(-1, int(steps) + 1):
            for inequal in ['lt','gt']:
                thresh_val = (range_min + float(j) * steps)
                predicted_val = stump_classify(data_matrix, i, thresh_val, inequal)
                error_array = mat(ones((m,1)))
                error_array[predicted_val == label_mat] = 0
                # 计算加权错误率
                weight_error = D.T * error_array
                print "split: dim %d, thresh %.2f, thresh ineqal:%s, the weight error is %.3f" % \
                      (i, thresh_val, inequal, weight_error)
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = predicted_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est



if __name__ == '__main__':
    D = mat(ones((5,1))/5)
    data_array, class_labels = loadSimpData()
    build_stump(data_array, class_labels, D)