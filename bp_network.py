#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets


if __name__=='__main__':
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()











