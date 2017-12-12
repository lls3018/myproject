# -*- coding:utf-8 -*-
__author__ = 'leon'
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot  as plt


# 设置恒纵轴字体大小
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

np.random.seed(42)

x = np.linspace(0, 5, 100)
y = 2*np.sin(x) + 0.3*x**2
y_data = y + np.random.normal(scale=0.3, size=100)
plt.figure('data')
plt.plot(x, y_data, '.')
plt.figure('model')
plt.plot(x, y)
plt.figure('data & model')
# k 制定颜色，lw制定宽度
plt.plot(x, y, 'k', lw=3)
plt.scatter(x, y_data)
plt.show()




