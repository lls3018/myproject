# -*- coding:utf-8 -*-
__author__ = 'leon'

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inx, data_set, labels, k):
    # 计算点到数据集中每个点的距离
    data_set_size = data_set.shape[0]
    diffMat = tile(inx, (data_set_size, 1)) - data_set
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def format_data_set(filename):
    with open(filename) as f:
        array_lines = f.readlines()
        number_lines = len(array_lines)
        data_mat = zeros((number_lines, 3))
        class_label_vector = []
        index = 0
        for line in array_lines:
            line  = line.strip()
            list_from_line = line.split('\t')
            data_mat[index,:] = list_from_line[0:3]
            class_label_vector.append(list_from_line[-1])
            index += 1
        return data_mat, class_label_vector


def auto_norm(data_set):
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    ranges = max_value - min_value
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = (data_set - tile(min_value, (m, 1))) / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_value


if __name__ == '__main__':
    # group, labels = create_data_set()
    # result = classify([0,0], group, labels, 3)
    # print "result",result
    # 数据集为[每年飞行的里程数，玩游戏所耗的时间比， 每周消费的冰激凌公升数]
    data_mat, labels = format_data_set('C:\Users\user\Desktop\work\projects\myproject\data\datingTestSet2.txt')
    auto_norm(data_mat)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(data_mat[:,1], data_mat[:,2])
    #plt.show()

