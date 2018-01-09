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
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

def calculate_error():
    pass

if __name__ == '__main__':
    image = None
    value = []
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
    error_list = calculate_error(out5, value)
    delta4 = F5.backward(error_list)
    delta3 = S4.backward(delta4)
    delta2 = C3.backward(delta3)
    delta1 = S2.backward(delta2)
    C1.backward(delta1)








