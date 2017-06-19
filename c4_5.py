# -*- coding:utf-8 -*-
__author__ = 'leon'

from math import log


def create_data_set():
    data_set = [[0, 0, 0, 0, 'N'],
                [0, 0, 0, 1, 'N'],
                [1, 0, 0, 0, 'Y'],
                [2, 1, 0, 0, 'Y'],
                [2, 2, 1, 0, 'Y'],
                [2, 2, 1, 1, 'N'],
                [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return data_set, labels


def create_test_set():
    test_set = [[0, 1, 0, 0],
                [0, 2, 1, 0],
                [2, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [2, 1, 0, 1]]
    return test_set


def calculate_shannon_ent(data_set):
    ent_num = len(data_set)
    labels_counts = {}
    for feature in data_set:
        current_label = feature[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        labels_counts[current_label] += 1
    shannon_ent = 0.0
    for key in labels_counts:
        # 计算每一种结果的概率
        prob = float(labels_counts[key]) / ent_num
        # 计算香浓熵
        shannon_ent -= prob*log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    # 去除某属性后分裂出的新的数据矩阵
    result = []
    for feature in data_set:
        if feature[axis] == value:
            # 指定数据从数据矩阵中去除
            reduce_feature = feature[:axis]
            reduce_feature.extend(feature[axis+1:])
            result.append(reduce_feature)
    return result


def choose_best_feature_to_split(data_set):
    feature_num = len(data_set[0]) - 1
    base_entropy = calculate_shannon_ent(data_set)
    # 最好的信息增益率
    best_info_gain_ratio = 0.0
    # 最好的分类属性
    best_feature = -1
    for i in range(feature_num):
        # 按顺序计算每一列数据的信息增益率，例如计算属性气温里的几种情况
        feature_list = [d[i] for d in data_set]
        unique_values = set(feature_list)
        new_entropy = 0.0
        split_info = 0.0
        for value in unique_values:
            # 气温里面包含，高，中，低值，每一个值的概率，和被选中时所包含的信息量
            # 气温为高时的子集
            sub_data_set = split_data_set(data_set, i, value)
            # 温度高的概率
            prob = len(sub_data_set) / float(len(data_set))
            # 气温为高时剩余未知的信息量
            new_entropy += prob*calculate_shannon_ent(sub_data_set)
            split_info += -prob*log(prob, 2)
        # 如果气温信息确定后，信息量减少了多少，信息增益
        info_gain = base_entropy - new_entropy
        if split_info == 0:
            # 当气温只有一种情况是，不能计算也不需要计算，这个属性没有意义
            continue
        # 计算信息增益率
        info_gain_ratio = info_gain / split_info
        if info_gain_ratio > best_info_gain_ratio:
            best_info_gain_ratio = info_gain_ratio
            best_feature = i
    return best_feature


def create_tree(data_set, labels):
    # 最后一列为类型列表
    class_list = [d[-1] for d in data_set]
    # 类型列表中只剩下相同类型时，停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 选择当前最佳划分节点属性
    best_feature = choose_best_feature_to_split(data_set)
    best_label = labels[best_feature]
    # print "best_label", best_label
    my_tree = {best_label: {}}
    del(labels[best_feature])
    # 获取气温的所有属性
    feature_values = [d[best_feature] for d in data_set]
    unique_values = set(feature_values)
    # print "unique_values",unique_values
    for value in unique_values:
        # 产生树子节点
        sub_labels = labels[:]
        # print "sub_labels", sub_labels
        # 递归产生树
        my_tree[best_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
    return my_tree


def classify(input_tree, labels, test_value):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feature_index = labels.index(first_str)
    for key in second_dict.keys():
        if test_value[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], labels, test_value)
            else:
                class_label = second_dict[key]
    return class_label


def classify_all(input_tree, labels, test_data_set):
    class_label_all = []
    for test_value in test_data_set:
        class_label_all.append(classify(input_tree, labels, test_value))
    return class_label_all


def main_bak():
    data_set, labels = create_data_set()
    labels_tmp = labels[:]
    decision_tree = create_tree(data_set, labels_tmp)
    # print "decision_tree", decision_tree
    test_set = create_test_set()
    # 验证数据
    result = classify_all(decision_tree, labels, test_set)
    print "result", result


def main():
    labels = ['buying', 'maintenance', 'doors', 'persons', 'lug_boot', 'safety']
    data_set = []
    with open('car_data') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            data_set.append(data)
    decision_tree = create_tree(data_set, labels)
    print "decision_tree", decision_tree


if __name__ == '__main__':
    main()
