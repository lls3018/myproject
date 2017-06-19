# -*- coding:utf-8 -*-
__author__ = 'leon'
import operator


def create_data_set():
    data_set = [[0,0,0,0,'N'],
                [0,0,0,1,'N'],
                [1,0,0,0,'Y'],
                [2,1,0,0,'Y'],
                [2,2,1,0,'Y'],
                [2,2,1,1,'N'],
                [1,2,1,1,'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return data_set, labels


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        else:
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reversed=True)
    return sorted_class_count


def split_data_set(data_set, axis, value):
    '''
    遍历数据集，将天气为冷的数据都过滤出来，并去掉天气这一选项返回
    '''
    result = []
    for feat in data_set:
        # 遍历每一天的数据
        if feat[axis] == value:
            # 当这一天的气温axis为 value冷时
            # 将这一天的其他数据加入过滤结果
            reduce_feat = feat[:axis]
            reduce_feat.extend(feat[axis+1:])
            result.append(reduce_feat)
    # result 的长度为数据集中，天气为冷的天数
    return result


def choose_best_feature_to_split(data_set):
    feature_num = len(data_set[0]) - 1
    best_gini = 999999.0
    best_feature = -1
    for i in range(feature_num):
        #  遍历每一种判断类型
        feat_list = [d[i] for d in data_set]
        # 选中每一种类型的所有情况
        unique_values = set(feat_list)
        # 把情况去重
        gini = 0.0
        for value in unique_values:
            # 选中其中一种情况 例如: i=气温，value=冷
            # 返回气温为冷的子集
            sub_data_set = split_data_set(data_set, i, value)
            # 计算气温为冷的概率
            prob = len(sub_data_set) / float(len(data_set))
            # 计算该子集中结果为N的概率
            sub_prob = len(split_data_set(sub_data_set, -1, 'N')) / float(len(sub_data_set))
            # 计算基尼指数Gini
            gini += prob*(1.0 - pow(sub_prob, 2) - pow(1 - sub_prob, 2))
        if (gini < best_gini):
            # 找出Gini指数最小的分类属性
            best_gini = gini
            best_feature = i
    print "best_gini",best_gini
    return best_feature


def create_tree(data_set, labels):
    # 取出分类列表
    class_list = [ d[-1] for d in data_set]
    # class_list = ['N','N','Y','Y','N','Y']
    if class_list.count(class_list[0]) == len(class_list):
        # 类别完全相同停止分类
        return class_list[0]
    if len(data_set[0]) == 1:
        # 遍历所有特征时反回出现的次数最多的
        return majority_count(class_list)
    best_feature = choose_best_feature_to_split(data_set)
    print "best_feature",best_feature
    best_feature_label = labels[best_feature]
    print "best_feature_label",best_feature_label
    my_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_value = [d[best_feature] for d in data_set]
    unique_values = set(feature_value)
    # 递归生成整个完整树
    for value in unique_values:
        sub_labels = labels[:]
        # 生成左右子树
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
    return my_tree


def main():
    data_set, labels = create_data_set()
    my_tree = create_tree(data_set, labels)
    print "my_tree", my_tree


if __name__ == '__main__':
    main()
