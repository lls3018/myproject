#coding:utf-8
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2


def logistic(x):
    return 1/(1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetwork(object):
    def __init__(self, layers, activation='tanh'):
        # layers=[3,3,2], 第一层3个元，第二层3个元，第三层2个元
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        # 初始化权重
        print "layers",layers
        for i in range(1, len(layers) - 1):
            print "11111111111",i,layers[i]
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
            print "2222222222",self.weights

    def fit(self, X, y, leaning_rate=0.2, epochs=10000):
        '''
        :param X: 矩阵，每行是一个实例
        :param y: 每个实例对应一个结果
        :param leaning_rate: 学习率
        :param epochs: 抽样方法对网络进行更新的最大次数
        :return:
        '''
        X = np.atleast_2d(X)
        print "X",X
        # 产生单位矩阵
        temp = np.ones([X.shape[0], X.shape[1]+1])
        print "temp",temp
        print "temp1",temp[:,0:-1]
        temp[:,0:-1] = X
        print "temp",temp
        X = temp
        print "X",X
        y = np.array(y)
        print "y",y
        for k in range(epochs):
            # 随机选一行进行更新
            i = np.random.randint(X.shape[0])
            print "X[i]",X[i]
            a = [X[i]]
            print "a",[X[i]]
            # 完成所有正向更新,更新权重
            for l in range(len(self.weights)):
                print "a[l]",a[l]
                print "self.weights[l]",self.weights[l]
                print "dot", np.dot(a[l], self.weights[l])
                a.append(self.activation(np.dot(a[l], self.weights[l])))
                print "a",a[-1]
                error = y[i] - a[-1]
                print "error",error
                deltas = [error * self.activation_deriv(a[-1])]
                print "deltas",deltas
                #反向计算误差，更新权重
                for l in range(len(a)-2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                    deltas.reverse()
                print "deltas",deltas
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += leaning_rate * layer.T.dot(delta)

    def predict(self, x):
        # 预测函数
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for j in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[j]))
            return a


if __name__=='__main__':
    nn = NeuralNetwork([2,2,1], 'tanh')
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    for i in range([[0,0],[0,1],[1,0],[1,1]]):
        print(i, nn.predict(i))












