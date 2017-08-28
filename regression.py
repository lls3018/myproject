# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *
import matplotlib.pyplot as plot


def loadDataSet(fileName):
    # 导入测试数据函数
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def stand_regress(XArr, YArr):
    # 标准回归函数
    X = mat(XArr); Y = mat(YArr).T
    XTX = X.T * X
    #矩阵求行列式，如果矩阵行列式为0，求矩阵逆将报错
    if linalg.det(XTX) == 0.0:
        return 0
    ws = XTX.I * (X.T * Y)
    return ws


def locally_weight_linear_regression(testPoint, xArr, yArr, k=1.0):
    # 局部加权线性回归
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 权重矩阵为对角矩阵，初始化为1
    weights = mat(eye((m)))
    # 遍历整个样本集，更新权重矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        # 样本点距离测试的点距离近的点权重越大，距离测试点越远点权重越小
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        return False
    ws = xTx.I*(xMat.T*weights*yMat)
    return testPoint*ws


def test_weight(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        #print "testArr[i]",testArr[i]
        #print "xArr",xArr
        #print "yArr",yArr
        yHat[i] = locally_weight_linear_regression(testArr[i], xArr, yArr, k)
    return yHat


def rssError(yArr, yHatArr):
    #print "(yArr-yHatArr)",(yArr-yHatArr)
    return ((yArr-yHatArr)**2).sum()


def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stage_wise(xArr, yArr, eps=0.01, numIt=100):
    '''
        eps 每次迭代需要的步长
        numIt 迭代次数
    '''
    # 向前逐步线性回归
    xMat = mat(xArr); yMat = mat(yArr).T
    # 计算y平均值
    yMean = mean(yMat, 0)
    # 计算y均差
    yMat = yMat - yMean
    # 标准化特征值
    xMat = regularize(xMat)
    m,n = shape(xMat)
    return_mat = zeros((numIt, n))
    # 初始化w系数
    ws = zeros((n,1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        # 对每一个系数做略微调整，更新系数为误差最小的w
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                # 变化第J个系数
                ws_test[j] += eps*sign
                # 计算变化系数后的预测值
                yTest = xMat*ws_test
                # 计算预测值与真实值的误差
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    # 更新最小误差和系数
                    lowestError = rssE
                    ws_max = ws_test
        ws = ws_max.copy()
        # 更新一行系数
        return_mat[i,:] = ws.T
    return return_mat


if __name__ == '__main__':
    xArr, yArr = loadDataSet('D:\\fang\work\projects\myproject\data\\abalone.txt')
    stage_wise(xArr, yArr, 0.01, 5000)
    '''
    abX, abY = loadDataSet('D:\\fang\work\projects\myproject\data\\abalone.txt')
    print "abY",abY
    yHat01 = test_weight(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = test_weight(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = test_weight(abX[100:199], abX[0:99], abY[0:99], 10)
    #print "yHat01",yHat01
    print rssError(abY[100:199], yHat01.T)
    print rssError(abY[100:199], yHat1.T)
    print rssError(abY[100:199], yHat10.T)
    '''
    '''
    xArr, yArr = loadDataSet('D:\\fang\work\projects\myproject\data\ex0.txt')
    yHat = test_weight(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    strInd = xMat[:,1].argsort(0)
    xSort= xMat[strInd][:,0,:]
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[strInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plot.show()
    '''
    '''
    xMat = mat(xArr); yMat = mat(yArr)
    YH = xMat * ws
    print corrcoef(YH.T, yMat)
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    YH = xCopy * ws
    ax.plot(xCopy[:,1], YH)
    plot.show()
    '''
