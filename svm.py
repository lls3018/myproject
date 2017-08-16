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
    # 如果a大于最大值，a取最大值
    # 如果a小于最小值，a取最小值
    # 让a不能超出范围
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
            #print "data_matrix*data_matrix[i:].T)",data_matrix[i,:]
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
                Ej = fXj - float(label_matrix[j])
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
                if (abs(alphas[j] - alpha_j_old) < 0.00001):
                    continue
                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_j_old - alphas[j])
                b1 = b - Ei - label_matrix[i] * (alphas[i] - alpha_i_old)*\
                    data_matrix[i,:]*data_matrix[i,:].T - \
                    label_matrix[j]*(alphas[j] - alpha_j_old)* \
                    data_matrix[j,:]*data_matrix[j,:].T
                b2 = b - Ej - label_matrix[i] * (alphas[i] - alpha_i_old)*\
                    data_matrix[i,:]*data_matrix[j,:].T -\
                    label_matrix[j]*(alphas[j] - alpha_j_old) *\
                    data_matrix[i,:]*data_matrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alpha_pairs_changed += 1
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas


class OptStruct:
    def __init__(self, dataMatin, classlabels, C, totel):
        self.X = dataMatin
        self.labelMat = classlabels
        self.C = C
        self.tol = totel
        self.m = shape(dataMatin)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2))) # 误差缓存


def calcEk(os, k):
    # 计算第K个样本的误差
    fEk = float(multiply(os.alphas, os.labelMat).T * (os.X * os.X[k,:].T)) + os.b
    Ei = fEk - float(os.labelMat[k])
    return Ei


def select_j(i, os, Ei):
    # 启发式方法
    # 选出其余样本中，与现样本中的误差步长最大的样本
    maxK = -1; maxDeltaE = 0; Ej = 0
    os.eCache[i] = [1, Ei]
    vaildEcacheList = nonzero(os.eCache[:, 0].A)[0]
    if (len(vaildEcacheList)) > 1:
        for k in vaildEcacheList:
            # 遍历误差缓存中数据
            if k == i:
                continue
            # 选出一个新的样本的误差值
            Ek = calcEk(os, k)
            # 计算新选出来的样本的误差与已选出来的样本误差步长
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                # 选出最大步长
                maxDeltaE = deltaE; maxK = k; Ej = Ek
        return maxK, Ej
    else:
        j = select_jrand(i, os.m)
        Ej = calcEk(os, j)
    return maxK, Ej


def update_Ek(os, k):
    # 计算误差并存入到缓存当中
    # 计算当前误差
    Ek =  calcEk(os, k)
    # 更新该样本在误差缓存中的误差
    os.eCache[k] = [1, Ek]


def inner_L(i, os):
    # 内循环函数
    # 计算制定样本的误差
    Ei = calcEk(os, i)
    # 如果误差超出范围，且α可以被优化
    if ((os.labelMat[i]*Ei < -os.tol) and (os.alphas[i] < os.C)) or\
            ((os.labelMat[i]*Ei > os.tol) and (os.alphas[i] > 0)):
        j, Ej = select_j(i, os, Ei)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        if os.labelMat[i] != os.labelMat[j]:
            # 如果这两个样本不在一个方向内
            # 下界
            L = max(0, os.alphas[j] - os.alphas[i])
            # 上界
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            # 如果这两个样本在一个方向内
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H: print "L == H"; return 0
        eta = 2.0 * os.X[i,:] * os.X[j,:].T - os.X[i,:] * os.X[i,:].T -  os.X[j,:] * os.X[j,:].T
        if eta >= 0: print "eta >= 0"; return 0
        # 计算出新的αj
        os.alphas[j] -= os.labelMat[j] * (Ei - Ej) / eta
        # αj 要在上下界之间
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)
        # 更新样本j误差缓存, 每次都是αi,αj一起优化
        update_Ek(os, j)
        if (abs(os.alphas[j] - alpha_j_old) < 0.00001):
            # 如果优化后αj的变化小于一定阈值，结束优化
            return 0
        # αj的变化大于一定值，之后，更新αi
        os.alphas[i] += os.labelMat[j] * os.labelMat[i] * (alpha_j_old -  os.alphas[j])
        # 更新样本i误差缓存
        update_Ek(os, i)
        # 更新b
        b1 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alpha_i_old) *\
            os.X[i,:] * os.X[i,:].T - os.labelMat[j]*\
            (os.alphas[j] -  alpha_j_old) * os.X[i,:] * os.X[j,:].T
        b2 = os.b - Ej - os.labelMat[i] * (os.alphas[i] - alpha_i_old) *\
            os.X[i,:] * os.X[j,:].T - os.labelMat[j]*\
            (os.alphas[j] -  alpha_j_old) * os.X[j,:] * os.X[j,:].T
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1 # 优化成功
    else:
        return 0 # 优化失败


def smo(dataMatIn, class_labels, C, toler, maxIter, kTup=('lin', 0)):
    '''
    :param dataMatIn: 数据集
    :param class_labels: 分类标签集合
    :param C: 弹性系数
    :param toler: 误差
    :param maxIter: 最大循环次数
    :param kTup:
    :return:
    '''
    # 先结构化数据集合
    os = OptStruct(mat(dataMatIn), mat(class_labels).transpose(), C, toler)
    iter = 0
    entireSet = True # true为遍历全集，false为只遍历非边界值
    alphaPairsChanged = 0
    while( iter < maxIter ) and ((alphaPairsChanged > 0) or (entireSet)):
        # 每次循开始时变化值都为0
        alphaPairsChanged = 0
        if entireSet:
            # 遍历所有值
            for i in range(os.m):
                # 遍历每一个样本进行优化，计算优化成功的样本数量
                alphaPairsChanged += inner_L(i, os)
                print u"循环次数:%d, 样本i:%d, 优化成功次数:%d" % (iter, i, alphaPairsChanged)
                iter += 1
        else:
            # 遍历所有非边界值
            nonBoundIs = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += inner_L(i, os)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            # 当遍历优化之后，变化值仍为0时，退出优化
            entireSet = False
    return os.b, os.alphas


def calculate_w(alphas, data, labels):
    # 计算出分类函数系数w
    X = mat(data)
    Y = mat(labels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*Y[i], X[i,:].T)
    return w


def test():
    x1 = np.arange(9.0).reshape((3,3))


if __name__ == '__main__':
    data_array, label_array = loadDataSet('C:\Users\user\Desktop\work\projects\myproject\data\SvmTestSet.txt')
    # smo 算法，通过不断优化逼近算出α和b,为进一步计算出超平面做准备
    b, alphas = smo(data_array, label_array, 0.6, 0.001, 40)
    ws = calculate_w(alphas, data_array, label_array)
    print "wwwwwwwwwwwww", ws
    # 计算样本的分类结果 y = wX + b
    dataMat = mat(data_array)
    print dataMat[0]*mat(ws) + b



