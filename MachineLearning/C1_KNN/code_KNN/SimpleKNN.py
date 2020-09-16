import numpy as np
import matplotlib.pyplot as plt


# k 近邻算法（KNN）
# 计算已知类别数据集中的点与当前点之间的距离；
# 按照距离递增次序排序；
# 选取与当前点距离最小的k个点；
# 确定前k个点所在类别的出现频率；
# 返回前k个点所出现频率最高的类别作为当前点的预测分类。


def crate_data_set():
    """ 生成数据集
    :return: 数据和标签
    """
    group = np.array([
        [1, 101], [5, 89], [108, 5], [115, 8]
    ])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


def classify0(test_set, dataSet, labels, k):
    """ 简单KNN算法实现
    :param test_set: 测试集
    :param dataSet: 训练集
    :param labels: 标签
    :param k:
    :return:
    """
    dataSetSize = dataSet.shape[0]
    A = np.tile(test_set, (dataSetSize, 1))
    diffMat = np.tile(test_set, (dataSetSize, 1)) - dataSet  # 矩阵运算的思想
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 搞清楚axis的含义 axis=1表示计算n(行）*m(列)得到n(行)*1列数据
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]  # 投票标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    dataSet, labels = crate_data_set()
    test = np.asarray([101, 20])
    test_class = classify0(test, dataSet, labels, 3)
    print("预测结果：" + test_class)
