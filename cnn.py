# -*- coding:utf-8 -*-
__author__ = 'leon'
import numpy
from theano.tensor.nnet import conv

'''
    0.输入28*28
    1.6个5x5卷积核，生成6个24x24图片
    2.2x2采样缩小一半，生成6个12x12图片
    3.全连接卷积，实际情况并不是全连接的，12组，一组6个5x5卷积核，生成12个8x8图片
    4.2x2采样缩小一半，生成12个4x4图像
    5.将12个4x4图像展开得到12*4*4=192维向量，输出层有10个神经元，每个神经元与192位相连
'''


def func(x):
    # 激活函数
    return x


def func_diff(x):
    # 激活函数导数
    return x*(1-x)

class ConvLayer1(object):
    '''
        接受输入28*28图像
        由6个5x5卷积核
        生成6个24x24图像
    '''
    def __init__(self, input, learning_rate):
        # 输入一张图片
        self.input = input
        # 学习率
        self.learning_rate = learning_rate
        # 本层卷积核
        self.kernel = [[]]*6
        # 本层偏置项
        self.bias = []
        # 6张图有6个灵敏度矩阵，每个像素有一个灵敏度值
        self.delta = [[]]*6

    def func(self, u):
        # 激活函数
        return u

    def forward(self):
        out = []
        for i in range(0, len(self.bias)):
            out.append(self.func(conv(self.input, self.kernel[i]) + self.bias[i]))
        return out

    def backward(self):
        # 计算偏置项梯度
        gradient_b = numpy.asarray(map(lambda x: numpy.sum(x), self.delta))
        # 计算卷积核梯度
        gradient_k = numpy.asarray(numpy.rot90(conv(self.input, numpy.rot90(self.delta, 2)), 2))
        # 更新参数
        self.bias -= self.learning_rate*gradient_b
        self.kernel -= self.kernel*gradient_k
        # 上一层是输入层，不需要继续传递灵敏度


class PoolLayer2(object):
    def __init__(self, input, delta):
        self.input = input
        # 本层灵敏度
        self.delta = delta

    def downsample(self, x):
        pass

    def forward(self):
        # 2x2采样缩小一半，生成6个12x12图片
        return numpy.asarray(map(lambda x: self.downsample(x), self.input))

    def backward(self):
        # 本层灵敏度已知，向前传递灵敏度
        pass


class ConvLayer3(object):
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class PoolLayer4(object):
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class FullLayer5(object):
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
        # 神经元个数为10
        self.nerve_num = 10
        # 输出为10位全连接神经元
        self.output = numpy.zeros(self.nerve_num)
        # 本层灵敏度
        self.delta = numpy.zeros(self.nerve_num)
        # 12*4*4*10=1920个参数
        self.w = numpy.random.rand(10, 12*4*4)
        # 10个神经元10个偏置项
        self.b = []
        # S4层输出值,被平铺后
        self.array_x = numpy.zeros(12*4*4)
        # 激活函数
        self.func = func
        self.func_diif = func_diff

    def forward(self, images):
        """
        :param images: 12张4x4图像
        """
        # 将所有图像展开一维192向量
        self.array_x = numpy.concatenate(map(lambda x: x.ravel(), images))
        for i in range(0, self.nerve_num):
            self.output[i] = self.func(numpy.dot(self.w[i],self.array_x)+self.b[i])

    def backward(self, value):
        # 计算误差
        error = value - self.output
        # 计算灵敏度
        for i in range(0, self.nerve_num):
            self.delta[i] = (-1)*func_diff(self.output[i])
        # 计算b梯度，最外层b梯度就等于灵敏度
        gradient_b = self.delta
        # 计算w梯度==灵敏度*S4层输出 12*4*4*10个梯度
        gradient_w = self.delta.T*self.output
        # 更新参数
        self.b -= self.learning_rate*gradient_b
        self.w -= self.learning_rate*gradient_w
        # 返回本层灵敏度
        return self.delta


if __name__ == '__main__':
    image = None
    value = numpy.asarray([0,1,2,3,4,5,6,7,8,9])
    C1 = ConvLayer1()
    S2 = PoolLayer2()
    C3 = ConvLayer3()
    S4 = PoolLayer4()
    F5 = FullLayer5()
    out1 = C1.forward(image)
    out2 = S2.forward(out1)
    out3 = C3.forward(out2)
    out4 = S4.forward(out3)
    out5 = F5.forward(out4)
    delta4 = F5.backward(value)
    delta3 = S4.backward(delta4)
    delta2 = C3.backward(delta3)
    delta1 = S2.backward(delta2)
    C1.backward(delta1)








