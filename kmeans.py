# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    '''
        计算两个向量的欧式空间距离
    :param vecA:
    :param vecB:
    :return:
    '''
    return sqrt(sum(power(vecA - vecB), 2))


def rand_cent(data_set, K):
    '''
        选出随机质心
    :return:
    '''
    n = shape(data_set)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        # 生成K个质心， 相当于K个堆
        min_J =  min(data_set[:,j])
        range_J = float(max(data_set[:,j]) - min_J)
        centroids[:,j] = min_J + range_J * random.rand(k,1)
    return centroids


def k_means(data_set, K, dist_means=distEclud, create_cent=rand_cent):
    '''
    :param data_set:
    :param K:
    :param dist_means:
    :param create_cent:
    :return:
    '''
    m = shape(data_set)[0]
    cluster_assment = mat(zeros((m,2)))
    # 产生K个随机质心
    centroids = create_cent(data_set, K)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        # 为样本集所有样本分配质心
        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(K):
                # 计算样本到质心的空间距离
                dist_JI = distEclud(centroids[j,:], data_set[i,:])
                # 寻找最近的质心
                if dist_JI < min_dist:
                    min_dist = dist_JI
                    min_index = j
            if cluster_assment[i,0] != min_index:
                # 最近的质心没有发生变化，退出
                cluster_changed = True
            # 存储样本i的最小质心和到质心的距离平方
            cluster_assment[i,:] = min_index, min_dist**2
        # 更新质心位置
        for cent in range(K):
            # 找出一个质心附近的所有点
            ptsInClust = data_set[nonzero(cluster_assment[:,0] == cent)[0]]
            # 求这些点的平均值，并更新该值为质心值
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, cluster_assment


def bin_kmeans(data_set, K, dist_means=distEclud):
    '''
        二分K-均值聚类算法
        将所有点看成一个簇
        当簇数目小于K时
        对每个簇
            计算总误差
            在给定的簇上进行K-均值聚类
            计算该簇一分为二之后的总误差
        选择哪个使误差最小的那个簇进行划分操作
    :param data_set:
    :param K:
    :param dist_means:
    :return:
    '''
    m = shape(data_set)[0]
    cluster_assment = mat(zeros((m,2)))


if __name__ == '__main__':
    data = loadDataSet('C:\Users\user\Desktop\work\projects\myproject\data\\testSet.txt')







