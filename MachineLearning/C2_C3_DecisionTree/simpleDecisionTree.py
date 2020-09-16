from math import log


def create_data_set():
    """创建数据集
    :return:
    """
    data_set = [[0, 0, 0, 0, 'no'],  # 数据集
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']  # 分类属性
    return data_set, labels  # 返回数据集和分类属性


def calc_shannon_entropy(dataSet):
    """计算香农熵
    :param dataSet:
    :return:
    """
    num_of_all = len(dataSet)
    label_counts = {}  # 计算一共有多少种标签

    for featVec in dataSet:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_of_all
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 轴
    :param value: 按照指定值划分
    :return: 被划分后的子集，子集用于方便计算香农熵
    """
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """计算信息增益
    :param dataSet:数据集
    :return:信息增益最大的特征索引
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calc_shannon_entropy(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calc_shannon_entropy(subDataSet)  # 根据公式计算子集的经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature


if __name__ == '__main__':
    data_set, features = create_data_set()
    print("最优特征索引值：" + str(chooseBestFeatureToSplit(data_set)))
