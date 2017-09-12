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
    header_table = {}
    for trans in data_set:
        for item in trans:
            # 生成元素出现次数字典列表{'a': 6, 'b': 3,...}
            header_table[item] = header_table.get(item, 0) + data_set[trans]
    for k in header_table.keys():
        # 过滤掉频率太低的元素
        if header_table[k] < min_support:
            del(header_table[k])
    # 生成频繁元素 {'a','b',...}
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:
        return None,None
    # 此时 header_table {'a': [6, None], 'b': [3, None]}
    for k in header_table:
        header_table[k] = [header_table[k], None]
    ret_tree = tree_node('NUll SET', 1, None)
    # 遍历所有数据集合
    for tran_set, count in data_set.items():
        local_D = {}
        # 遍历每个集合中每个元素
        for item in tran_set:
            # 判断该元素是否是频繁元素
            if item in freq_item_set:
                # 获取频繁元素的频率 {'a':6, 'b':3, ...}
                local_D[item] = header_table[item][0]
        # 如果该条数据集合的频繁元素个数大于0
        if len(local_D) > 0:
            # 对该条数据集合中的元素进行排序，次数最高的在前面, ['a', 'b',...]
            order_items = [v[0] for v in sorted(local_D.items(), key=lambda p:p[1], reverse=True)]
            #将该条集合插进树中
            update_tree(order_items, ret_tree, header_table, count)
    return ret_tree, header_table



def update_tree(items, in_tree, header_table, count):
    '''
    :param items: ['a','b',...]
    :param in_tree:
    :param header_table: {'a':[6,None], 'b':[3,None],...}
    :param count:
    :return:
    '''
    # items[0] == 'a'
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        # 生成新枝
        in_tree.children[items[0]] = tree_node(items[0], count, in_tree)
    # header_table['a']==None, children['a']
    if header_table[items[0]][1] == None:
        header_table[items[0]][1] = in_tree.children[items[0]]
    else:
        # 更新头指针表中制定元素节点
        update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    '''
    :param node_to_test: {'a'}[None]制定元素的节点列表
    :param target_node:
    :return:
    '''
    # 找到节点列表中最后节点
    while(node_to_test.node_link != None):
        node_to_test = node_to_test.node_link
    # 向最后节点添加新的节点
    node_to_test.node_link = target_node


def ascend_tree(leaf_node, prefix_path):
    '''
    从叶子节点向上追溯，直到根节点，找出这条路径
    :param leaf_node:
    :param prefix_path:
    :return:
    '''
    if leaf_node.parent != None:
        prefix_path.append(leaf_node)
    ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, tree_node):
    '''
    :param base_pat:
    :param tree_node:
    :return:
    '''
    cond_pats = {}
    while tree_node != None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats


if __name__ == '__main__':
    '''
    使用FP树高效发现频繁项集
    优点：速度快，创建完成树之后，不再需要原始数据集了,更快的发现频繁项集
    1.先创建一颗FP树{
        1.遍历数据集获得每个元素的出现频率
        2.缺掉不满足最小支持度元素项
        3.读入每个项集，将其添加到一条已经存在的路径中，如果该路径不存在创建一条新的

    }
    2.创建条件FP树{
        1.找到条件模式基:是以查找元素项为结尾的路径集合，例如以元素'a'为结尾的路径集合
    }
    '''
    root_node = tree_node('pyramid', 9, None)
    root_node.children['eye'] = tree_node('eye', 13, None)
    root_node.display()
    # 先进行排序
    #



