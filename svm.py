# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def select_jrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_sample(data_mat_in, class_labels, C, error_tolerant, max_inter):
    '''
    :param data_mat_in:  数据集
    :param class_labels: 类别标签
    :param C: 常数C
    :param error_tolerant: 容错率
    :param max_inter: 最大循环次数
    :return:
    '''
    # 创建一个初始alpha向量并将初始化为0的向量
    # 当迭代次数小于最大迭代次数时
        # 对数据集中的每一个数据向量
            # 如果该数据向量可以被优化
                # 随机选择另外一个数据变量
                # 同时优化这两个向量
                # 如果两个向量都不能被优化，退出循环
        # 如果所有向量都不能被优化，增加迭代数目，继续下一循环
    data_matrix = mat(data_mat_in)
    #print 'data_matrix',data_matrix
    label_matrix= mat(class_labels).transpose()
    #print "label_matrix",label_matrix
    #print "class_labels",class_labels
    #print "label_mat",label_mat
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    #print "alphas",alphas
    iter = 0
    while (iter < max_inter):
        alpha_pairs_changed = 0
        for i in range(m):
            #print "alphas",alphas
            print "data_matrix*data_matrix[i:].T)",data_matrix[i,:]
            # 目标函数
            fXi = float(multiply(alphas, label_matrix).T * (data_matrix*data_matrix[i,:].T)) + b
            #print "fXi",fXi
            # 误差
            Ei = fXi - float(label_matrix[i])
            # α满足优化条件时进行优化
            if ((label_matrix[i]*Ei < -error_tolerant) and (alphas[i] < C)) or \
                    ((label_matrix[i]*Ei > error_tolerant) and alphas[i] > 0):
                # 随机选出另一个α与它进行配对优化，因为∑αy = 0 这个约束条件所以必须一对一对优化否则总和不等于0
                j = select_jrand(i, m)
                # 目标函数j的表达式
                fXj = float(multiply(alphas, label_matrix).T * (data_matrix*data_matrix[j,:].T)) + b
                # 计算其误差
                Ej = fXi - float(label_matrix[j])
                # 取出两个系数α
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # 保证α 在0-C之间
                if (label_matrix[i] != label_matrix[j]):
                    # 如果两个参数都在超平面一个方向, L为α 最小集合中的最大值，H为α最大集合中的到最小值
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] + C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    # 不懂
                    print "L==H"
                    continue
                #
                eta = 2.0 * data_matrix[i,:] * data_matrix[j,:].T - \
                    data_matrix[i,:] * data_matrix[i,:].T - \
                    data_matrix[j,:] * data_matrix[j,:].T
                if eta > 0:
                    continue
                alphas[j] -= label_matrix[j]*(Ei-Ej)/eta
                alphas[j] = clip_alpha(alphas[j], H, j)
                if (abs(alphas[j] - alpha_j_old)     < 0.00001):
                    continue
                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_j_old - alphas[j])
                b1 = b - Ei - label_matrix[i] * (alphas[i] - alpha_i_old)*\
                    data_matrix[i,:]*data_matrix[i,:].T - \
                    label_matrix[j]*(alphas[j] - alpha_j_old)* \
                    data_matrix[j,:]*data_matrix[j,:].T
                b2 = b-Ej-label_matrix[i]*(alphas[i] - alpha_i_old)*\
                    data_matrix[i,:]*data_matrix[j,:].T - \
                    label_matrix[j,:]*data_matrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[i]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alpha_pairs_changed += 1
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas



if __name__ == '__main__':
    data_array, label_array = loadDataSet('C:\Users\user\Desktop\work\projects\myproject\data\SvmTestSet.txt')
    #print "data_array",data_array
    smo_sample(data_array, label_array, 0.6, 0.001, 40)




