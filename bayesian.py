# -*- coding:utf-8 -*-
__author__ = 'leon'

from numpy import *


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0,1,0,1,0,1]
    return posting_list, class_vector


def create_vocab_list(data_set):
    # 获取去重词汇集合
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_words_to_vector(vocab_list, input_set):
    '''
    :param vocab_list: 文档矩阵中非重的词汇集
    :param input_set: 文档矩阵中的一行
    :return: 该行出现的词汇在词汇集中标识为1
    '''
    return_vector = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] = 1
    return return_vector


def trans_bayes(trans_matrix, train_category):
    '''
    :param trans_matrix: 数据矩阵
    :param train_category: 分类结果
    :return:
    '''
    docs_num = len(trans_matrix)
    words_num = len(trans_matrix[0])
    # 计算整片文档矩阵属于侮辱性文档(每一行为一个文档)的概率
    p_abusive = sum(train_category) / float(len(train_category))
    p0_num = p1_num = zeros(words_num)
    p0_denom = p1_denom = 0.0
    # 遍历整个文档矩阵
    for i in range(docs_num):
        if train_category[i] == 1:
            # 如果该文档属于侮辱文档，向量相加
            p1_num += trans_matrix[i]
            # 侮辱性文档中的词汇量
            p1_denom += sum(trans_matrix[i])
            print "p1_num",p1_num
            print "sum(trans_matrix[i])",sum(trans_matrix[i])
        else:
            # 非侮辱性文档，向量相加
            p0_num += trans_matrix[i]
            # 非侮辱性文档中的词汇量
            p0_denom += sum(trans_matrix[i])
    #
    p1_vector = p1_num / p1_denom
    print "sum(p1_vector)",sum(p1_vector)
    p0_vector = p0_num / p0_denom
    return p0_vector, p1_vector, p_abusive


if __name__ == '__main__':
    # 获取文档矩阵和每一行属性
    posts_lists, class_list = load_data_set()
    # 获取文档中词汇集合，不重复
    vocab_list = create_vocab_list(posts_lists)
    # 将文档词汇矩阵转化为向量矩阵
    train_mat = []
    for post in posts_lists:
        train_mat.append(set_words_to_vector(vocab_list, post))
    p0, p1, pab = trans_bayes(train_mat, class_list)
    #print p0, p1, pab










