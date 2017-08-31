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
        ops 控制函数的停止时机,误差和几点最少样本数
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


def is_tree(obj):
    # 判断输入变量是否是一个树
    return (type(obj).__name__ == 'dict')


def get_mean(tree):
    '''
        # 从上到下遍历树直到叶子节点
        # 如果有两个叶节点，计算他们的平均值
    '''
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, test_data):
    '''
        tree 待剪枝树
        对树进行剪枝
        计算将两个叶子节点合并后的误差
        计算不合并的误差
        如果合并能降低误差，就合并
    '''
    if shape(test_data)[0] == 0:
        return get_mean(tree)
    if (is_tree(tree['left'])) or (is_tree(tree['right'])):
        left_set, right_set = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree('left'), left_set)
    if is_tree(tree['right']):
        tree['right'] = prune(tree('right'), right_set)
    # 当左右两边都不是树，是叶子节点的时候
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        left_tree, right_tree = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
        error_no_merge = sum(power(left_set[:,-1] - tree['left'], 2)) + sum(power(right_tree[:,-1] - tree['right'], 2))
        tree_mean = (tree['left']+tree['right'])/2.0
        error_merge = sum(power(test_data[:,-1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print "merge"
            return tree_mean
        else:
            return tree
    else:
        return tree


def linear_solve(data_set):
    '''
    '''
    m,n = shape(data_mat)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:, 1:n] = data_set[:,0:n-1]
    Y = data_set[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def model_leaf(data_set):
    '''
    '''
    ws, X, Y = linear_solve(data_set)
    return ws


def model_error(data_set):
    '''
    '''
    ws, X, Y = linear_solve(data_set)
    yHat = X * ws
    return sum(power((Y - yHat),2))


def reg_trees_eval(model, in_data):
    return float(model)


def model_tree_eval(model, in_data):
    n = shape(in_data)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = in_data
    return float(X*model)


def tree_fore_cast(tree, in_data, model_eval):
    '''
        输入单个数据或者行向量，返回浮点型
    '''


if __name__ == '__main__':
    data = loadDataSet('C:\Users\user\Desktop\work\projects\myproject\data\ex00.txt')
    data_mat = mat(data)
    create_tree(data_mat)


