# -*- coding:utf-8 -*-
__author__ = 'leon'
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot  as plt


a = np.array([1,2,3,4,5], ndmin=3)
a = np.array([[1,2,3],[4,5,6]])
a.shape = (3,2)
a = np.empty([3,2], dtype=int)
a = np.zeros(5)
a = np.ones([1, 10], dtype=int)
# 列表转数组
x = [1,2,3,4]
a = np.asarray(x)
a = np.arange(5)
a = np.arange(10, 20, 2)
a = np.linspace(10, 20, 5)
a = np.linspace(10, 20, 5, endpoint=False)
a = np.arange(10)
s = slice(2, 7, 2)
#print a[2:7:2]
#print a[s]
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
#print a
#print a[...,1] # 返回第二列元素
#print a[1,...] # 返回第二行元素
#print a[...,1:] # 返回第二列之后所有元素
x = np.array([[1,2],[3,4],[5,6]])
#print x
y = x[[0,1,2],[1,1,1]]
#print y
x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#print x
rows = np.array([[0,0],[3,3]])
#print rows
cols = np.array([[0,2],[0,2]])
#print cols
y = x[rows,cols]
#print(y)
a = np.arange(0,60,5)
#print a
a = a.reshape(3,4)
#print a
#print a.T
#for x in np.nditer(a):
#    print x,
#for x in np.nditer(a, flags=['external_loop'], order='F'):
#    print x,
#a = np.arange(12).reshape(3,4)
#print a
#print np.transpose(a)
#a = np.arange(9)
#print np.split(a,3)
# 按位置切割
#print np.split(a, [4,7])
#a = np.array([[1,2,3],[4,5,6]])


'''
x = np.arange(1,11)
y = 2*x + 5
plt.title('demo')
plt.xlabel('x axis caption')
plt.xlabel('y axis caption')
plt.plot(x, y, 'ob')
plt.show()
'''
'''
x = np.arange(0, 3*np.pi, 0.1)
y = np.sin(x)
plt.title('sine wave form')
plt.plot(x, y)
plt.show()
'''




#print np.sqrt(3)
a = np.random.randn(2, 3)
b = np.random.randn(200, 2)

print np.dot(b, a)






















