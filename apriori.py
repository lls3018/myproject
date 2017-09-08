# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    '''
    构建最底层集合{[1],[2],[3]....}
    '''
    c1 = []
    for transaction in data_set:
        # 遍历所有购物清单
        for item in transaction:
            # 遍历该购物清单中的所有商品
            if not [item] in c1:
                # 创建一个只包含一样商品的集合列表
                c1.append([item])
    c1.sort()
    return map(frozenset, c1)


def scan_data(D, Ck, min_support):
    '''
    淘汰候选集合中支持度低的集合
    :param D: 数据集
    :param Ck: 候选集合CK
    :param min_support: 最小支持度
    :return:符合条件的集合，符合条件集合的支持度
    '''
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    num_items = float(len(D))
    ret_list = []
    support_data = {}
    for key in ssCnt:
        # 计算所有集合的支持度
        support = ssCnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(Lk, k):
    '''
    输入[{1},{2},{3}],输出{{1,2},{1,3},{2,3}}
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return: 创建候选集 CK
    '''
    ret_list = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i+1, len_Lk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                # 将前K-2项相同的两个集合合并
                ret_list.append(Lk[i] | Lk[j])
    return ret_list


def apriori(data_set, min_support=0.5):
    C1 = create_c1(data_set)
    D = mat(set, data_set)
    # 淘汰候选集合中支持度低的集合
    L1, support_data = scan_data(D, C1, min_support)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        # 扫描数据集，从Ck到Lk
        Ck = apriori(L[k-2], k)
        Lk, supK = scan_data(D, Ck, min_support)
        support_data.update(supK)
        L.append(Lk)
        k += 1
    return L, support_data



if __name__=='__main__':
    pass








