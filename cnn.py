# -*- coding:utf-8 -*-
__author__ = 'leon'
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

def cnn():
    # 面向流程
    # 0.输入28*28
    sample_data = load_sample()
    # 6个5x5卷积核，生成6个24x24图片
    kernels = [[],[],[],[],[],[]]
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
    # 反向传递参数


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






