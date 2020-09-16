from math import log
import operator
import simpleDecisionTree


def createDataSet():
    """创建测试数据集
    :return:
        dataSet 数据集
        labels 特征标签
    """
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
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
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels  # 返回数据集和分类属性


def majorityCnt(classList):
    """统计classList中出现次数最多的元素（类标签）
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys()(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), ker=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def creatTree(dataSet, labels, featLabels):
    """创建决策树
    方法：递归创建
    :param dataSet:
    :param labels:
    :param featLabels:存储选择的最优特征标签
    :return:myTree 决策树
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 如果标签全部相等 则返回
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)  # 数据纬度不够
    bestFeat = simpleDecisionTree.chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = creatTree(simpleDecisionTree.splitDataSet(dataSet, bestFeat, value), labels,
                                                 featLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = creatTree(dataSet, labels, featLabels)
    print(myTree)
