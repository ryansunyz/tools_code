from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus
from sklearn.preprocessing import LabelEncoder


def C3_step1():
    """第三节课第1步
    :return:
    """
    with open('./data/lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lencesLables = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(lenses, lencesLables)


def C3_step2():
    """生成pandas数据 方便序列化工作
    :return:
    """
    with open('./data/lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_lable in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_lable)])
        lenses_dict[each_lable] = lenses_list
        lenses_list = []
    print(lenses_list)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)


def C3_step3():
    """对决策树可视化
    :return:
    """
    with open('./data/lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_lable in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_lable)])
        lenses_dict[each_lable] = lenses_list
        lenses_list = []
    print(lenses_list)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")


if __name__ == '__main__':
    C3_step3()
