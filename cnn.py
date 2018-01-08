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
    return x

# 卷积一层的数据结构
c1 = {
    # 6个卷积核5x5
    'num': 6,
    'kernel': [[1],[2],[3],[4],[5],[6]],
    'bias': [1,2,3,4,5,6]
}

class Conv1(object):
    def __init__(self):
        # 卷积一层数据
        self.params = {
            'num': 6,
            'kernel': [[1],[2],[3],[4],[5],[6]],
            'bias': [1,2,3,4,5,6],
            'u': [] # 未经过激活函数的数据，之后进行敏感度计算需要
        }
        # 本层灵敏度，由下一层计算得到，赋值给本层
        self.delta = None

    def func(self, u):
        # 激活函数
        return u

    def func_diff(self):
        # 激活函数求导
        return (1-self.params['u'])*self.params['u']

    def forward(self, input):
        out = []
        for i in range(0, self.params['num']):
            u = conv(input, self.params['kernel'][i]) + self.params['bias'][i]
            self.params['u'].append(u)
            out.append(self.func(u))
        return out

    def backward(self):
        # 根据本层灵敏度计算梯度



def conv1(sample_data, kernels, bias):
    # 每次输入与一个卷积核进行卷积操作
    out_list = []
    for k in kernels:
        out_list.append(func(con(sample_data, k, bias, func)))
    return out_list

def conv2(inputs, kernels, bias):
    # 全连接卷积，实际情况并不是全连接的，12组，一组6个5x5卷积核，生成12个8x8图片
    out_list = []
    # 每一次运算一组kernel
    for ks in kernels:
        # 每次一个卷积核与一个输入图像进行卷积
        out = None
        for i in range(0, len(inputs)):
            out += con(inputs[i], ks[i])
        out += bias[i]
        out_list.append(func(out))
    return out_list


def pool1(inputs):
    # 池化操作
    out_list = []
    for i in inputs:
        out_list.append(pool(i, 2))
    return out_list

def last_out(inputs, wights, bias):
    """
    将12个4x4图像展开得到12*4*4=192维向量，输出层有10个神经元，每个神经元与192位相连
    :param inputs: 192位向量
    :param wights: 10组192位参数
    :param bias: 10个偏重
    :return: 10位向量，表示预测0-9的概率
    """
    result = []
    for i in range(0, len(wights)):
        result.append(func(reduce(lambda x,y: x+y, inputs*wights[i])+bias[i]))
    return result

def update_s4(sens_list):
    # o5层有10个神经元，s4层


def cnn():
    # 面向流程
    # 0.输入28*28
    sample_data = load_sample()
    # 6个5x5卷积核，生成6个24x24图片
    kernels = [[],[],[],[],[],[]]
    # 卷积一层的数据结构

    bias = []
    out1 = conv2(sample_data, kernels, bias)
    # 2x2采样缩小一半，生成6个12x12图片
    out2 = pool1(out1)
    # 全连接卷积，实际情况并不是全连接的，12组，一组6个5x5卷积核，生成12个8x8图片
    kernels2 = []
    bias2 = []
    out3 = conv2(out2, kernels2, bias2)
    out4 = pool1(out3)
    # 将12个4x4图像展开得到12*4*4=192维向量，输出层有10个神经元，每个神经元与192位相连
    # 首先将数据进行展开
    y = last_out(out4)
    # 然后进行误差计算,进行反向传到
    error = loss(b, y)
    # 计算o5层，各个神经元的灵敏度准备向前传播，更新o5十个神经元的w和b
    o5_sen_list = update_o5()
    # 十个神经元十个敏感度，对应s4层的每个元素都有贡献
    # s4层有12*4*4个像素，相当于每个像素都有敏感度
    update_s4(o5_sen_list)





class ConLayer(object):
    '''
        卷积层
    '''
    def __init__(self):
        pass

    def conv(self, data, params):
        out_list = []
        for p in params:
            out_list.append(func(conv(data, p['kernel'], p['bias'])))
        return out_list

    def gradient(self):
        pass



class PoolingLayer(object):
    '''
        池化层
    '''
    def __init__(self):
        pass

    def mean(self, input):
        out = func( mean(input()) * w + b )
        return out


if __name__=='__main__':
    p_data = []
    # 参数需要初始化
    params = [{'kernel': [[1,1],[1,1]],'bias': []}]
    con = ConLayer()
    pool = PoolingLayer()
    out = con.conv(p_data, params)
    out = pool.mean(out)






