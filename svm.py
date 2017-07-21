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
            fXi = float(multiply(alphas, label_matrix).T * (data_matrix*data_matrix[i,:].T)) + b
            #print "fXi",fXi
            #
            Ei = fXi - float(label_matrix[i])
            # 如果alpha可以更改进入优化过程
            if ((label_matrix[i]*Ei < -error_tolerant) and (alphas[i] < C)) or \
                    ((label_matrix[i]*Ei > error_tolerant) and alphas[i] > 0):
                # 随机选出第二个alphas
                j = select_jrand(i, m)
                fXj = float(multiply(alphas, label_matrix).T * (data_matrix*data_matrix[j,:].T)) + b
                Ej = fXi - float(label_matrix[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()



        iter += 1





if __name__ == '__main__':
    data_array, label_array = loadDataSet('C:\Users\user\Desktop\work\projects\myproject\data\SvmTestSet.txt')
    #print "data_array",data_array
    smo_sample(data_array, label_array, 0.6, 0.001, 40)




