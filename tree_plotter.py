# -*- coding:utf-8 -*-
__author__ = 'leon'


import matplotlib.pyplot as plt


decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(my_tree):
    leaf_num = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            leaf_num += get_num_leafs(second_dict[key])
        else:
            leaf_num += 1
    return leaf_num


def get_tree_depth(my_tree):
    max_depth = 0
    firstStr = list(my_tree.keys())[0]
    second_dict = my_tree[firstStr]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = get_tree_depth(second_dict[key]) + 1
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_mid_text(cntrPt, parentPt, txt_string):
    x_mid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    y_mid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    create_plotter.ax1.text(x_mid, y_mid, txt_string)


def plot_node(node_txt, centerPt, parentPt, node_type):
    create_plotter.ax1.annotate(node_txt, xy=parentPt, xycoords='axes fraction',\
                                xytext=centerPt, textcoords='axes fraction',\
                                va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def plot_tree(my_tree, parentPt, node_txt):
    leaf_num = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntrpt = (plot_tree.xOff + (1.0 + float(leaf_num)) / 2.0 / plot_tree.totalw, plot_tree.yOff)
    plot_mid_text(cntrpt, parentPt, node_txt)
    plot_node(first_str, cntrpt, parentPt, node_txt)
    secondDict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plot_tree(secondDict[key], cntrpt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalw
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrpt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrpt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plotter(my_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plotter.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalw = float(get_num_leafs(my_tree))
    plot_tree.totalD = float(get_tree_depth(my_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalw
    plot_tree.yOff = 1.0
    plot_tree(my_tree, (0.5, 1.0), '')
    plt.show()
