# -*- coding:utf-8 -*-
__author__ = 'leon'


from numpy import *
arr = random.rand(4,4)
arr1 = mat(arr)
#print "arr1",arr1
#print eye(4)
#print mean(arr1)
data = mat([1,2,3,4])
#print data
#print data.A
#a = inf
#print var(data)

# sum(data) 对所有数据求和
# mean(data) 对所有元素求平均值
# mean(data, 0), 压缩行对列求平均值
# mean(data, 1), 压缩列对行求平均值
# var(data) 求方差 S =  1/(N-1)*∑(Xi - mean(X))**2, 计算样本每个值的方差
# power(data, num) num次方 power(5,2) == 25
import numpy as np
#print np.dot([1,2,3], [4,5,6])
#print np.ones(5)
print np.random.random((3,3))


