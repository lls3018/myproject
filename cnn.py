# -*- coding:utf-8 -*-
__author__ = 'leon'

def func(x):
    return x*x


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






