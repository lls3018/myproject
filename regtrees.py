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


def reg_leaf(data_set):
    return mean(data_set[:,-1])


def reg_error(data_set):
    # 计算子树的方差值
    return var(data_set[:, -1]) * shape(data_set)[0]


def bin_split_data_set(data_set, feature, value):
    '''
        将数据集根据特性切分得到两个子集
        按照哪个特性feature进行切分，阈值为value
    '''
    #print "data_set[:,feature] > value",data_set[nonzero(data_set[:,feature] <= value)[0], :][0]
    mat_0 = data_set[nonzero( data_set[:,feature] > value )[0], :]
    mat_1 = data_set[nonzero( data_set[:,feature] <= value )[0], :]
    return mat_0, mat_1


def choose_best_split(data_set, leaf_type=reg_leaf, error_type=reg_error, ops=(1,4)):
    '''
        找出最佳的分类方式，找到了，返回分列特性index和分列特性value
        如果找不到返回None并调用create_tree 生成叶子节点
        ops 控制函数的停止时机
    '''
    tolS = ops[0]; tolN = ops[1]
    # 如果所有值都相同退出
    if len(set(data_set[:,-1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m,n = shape(data_set)
    # 样本整体方差值
    S = error_type(data_set)
    best_S = inf; best_index = 0; best_value = 0
    # 遍历所有特征
    for feature_index in range(n-1):
        # 遍历样本中该特征
        for split_value in set(data_set[:, feature_index]):
            # 尝试用一个样本的指定特征值进行测试分类
            mat_0, mat_1 = bin_split_data_set(data_set, feature_index, split_value)
            if (shape(mat_0) < tolN) or (shape(mat_1) < tolN):
                # 产生的叶子节点中样本太少，此分类不合理，跳过
                continue
            # 计算此种分类，两边子树的误差
            new_S = error_type(mat_0) + error_type(mat_1)
            if new_S < best_S:
                # 取误差最小的特征作为分列特征
                best_index = feature_index
                best_value = split_value
                best_S = new_S
    if (S - best_S) < tolS:
        # 如果找到的最佳切分方法，误差降低太少，那么就不应该再进行切分了，而直接创建叶子节点
        return None, leaf_type(data_set)
    # 按照找到的最佳切分方法进行切分
    mat_0, mat_1 = bin_split_data_set(data_set, best_index, best_value)
    if (shape(mat_0)[0] < tolN) or (shape(mat_1)[0] < tolN):
        # 如果分列的左右子树中，有一个子树的样本数量太小，分列的意义就不大
        return None, leaf_type(data_set)
    return best_index, best_value


"""
def create_tree(data_set, leaf_type=reg_leaf, error_type=reg_error, ops=(1,4)):
    '''
    # 找到最佳待切分特征：
        # 如果该节点不能再分，该节点存为叶子节点
        # 执行二元切分
        # 在左子树上调用create_tree 方法
        # 在右子树上调用create_tree 方法
    '''
    feature, value = choose_best_split(data_set, leaf_type, error_type, ops)
    if feature == None:
        return value
    ret_tree = {}
    ret_tree['spInd'] = feature
    ret_tree['spVal'] = val
    left_set, right_set = bin_split_data_set(data_set, feature, value)
    ret_tree['left'] = create_tree(left_set, leaf_type, error_type, ops)
    ret_tree['right'] = create_tree(right_set, leaf_type, error_type, ops)
    return ret_tree
"""

if __name__ == '__main__':
    test_mat = mat(eye(4))
    mat0, mat1 = bin_split_data_set(test_mat, 1, 0.5)
    print mat0,mat1










