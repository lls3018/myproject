# -*- coding:utf-8 -*-
__author__ = 'leon'
from numpy import *


class tree_node():
    def __init__(self, name_value, num_occur, parent_node):
        self.name = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self, num_occur):
        # 对count变量增加给定值
        self.count += num_occur

    def display(self, ind=1):
        # 将树以文本形式显示
        print ' '*ind, self.name, '', self.count
        for child in self.children.values():
            child.disp(ind+1)


def create_tree(data_set, min_support=1):
    '''
    1.需要遍历数据两次，一次扫描所有元素出现的频率，
    :param data_set:
    :param min_support:
    :return:
    '''
    # {'a': 6, 'b': 3,...}
    header_table = {}
    for trans in data_set:
        for item in trans:
            # 生成元素出现次数字典列表
            header_table[item] = header_table.get(item, 0) + data_set[trans]
    for k in header_table.keys():
        # 过滤掉频率太低的元素
        if header_table[k] < min_support:
            del(header_table[k])
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:
        return None,None
    for k in header_table:
        header_table[k] = [header_table[k], None]
    ret_tree = tree_node('NUll SET', 1, None)
    # 遍历数据集合, 填充树
    for tran_set, count in data_set.items():
        local_D = {}
        for item in tran_set:
            if item in freq_item_set:
                local_D[item] = header_table[item][0]
        if len(local_D) > 0:
            # 对一条数据集进行元素排序
            order_items = [v[0] for v in sorted(local_D.items(), key=lambda p:p[1], reverse=True)]
            # 使用排序后的数据对树进行填充
            update_tree(order_items, ret_tree, header_table, count)
    return ret_tree, header_table



def update_tree(items, in_tree, header_table, count):
    '''
    :param items:
    :param in_tree:
    :param header_table:
    :param count:
    :return:
    '''
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        # 生成新枝
        in_tree.children[items[0]] = tree_node(items[0], count, in_tree)
    if header_table[items[0]][1] == None:
        header_table[items[0]][1] = in_tree.children[items[0]]



def update_header(node_to_test, target_node):
    '''
    :param node_to_test:
    :param target_node:
    :return:
    '''

if __name__ == '__main__':
    '''
    优点：速度快，创建完成树之后，不再需要原始数据集了,更快的发现频繁项集
    1.先创建一颗FP树{
        1.遍历数据集获得每个元素的出现频率
        2.缺掉不满足最小支持度元素项
        3.读入每个项集，将其添加到一条已经存在的路径中，如果该路径不存在创建一条新的

    }
    2.根据树挖掘出频繁项集
    '''
    root_node = tree_node('pyramid', 9, None)
    root_node.children['eye'] = tree_node('eye', 13, None)
    root_node.display()
    # 先进行排序
    #



