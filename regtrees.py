# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *
from Tkinter import *
import matplotlib.pyplot as plot


class TreeNode(object):
    def __init___(self, feature, value, right, left):
        feature_to_split_on = feature
        value_of_split = value
        right_branch = right
        left_branch = left


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def bin_split_data_set(data_set, feature, value):
    '''
        将数据集根据特性切分得到两个子集
    '''
    mat_0 = data_set[nonzero(data_set[:,feature]) > value][0,:][0]
    mat_1 = data_set[nonzero(data_set[:,feature]) <= value][0,:][0]
    return mat_0, mat_1


def choose_best_split():
    pass


def create_tree(data_set, leaf_type=reg_leaf, error_type=reg_error, ops=(1,4)):
    '''
    # 找到最佳待切分特征：
        # 如果该节点不能再分，该节点存为叶子节点
        # 执行二元切分
        # 在左子树上调用create_tree 方法
        # 在右子树上调用create_tree 方法
    '''
    feature, value = choose_best_split()
    if feature == None:
        return value
    return_tree = {}



if __name__ == '__main__':
    print "1111111111111"
