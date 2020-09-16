"""
拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，它就是为了解决0概率问题。
以及解决下溢出的问题
"""

import numpy as np
import codeNaiveBayes


def trainNB0(trainMatrix, trainCategory):
    """经过Laplas Smoothing的朴素贝叶斯训练
    :param trainMatrix:
    :param trainCategory:
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0;
    # TODO 不是很理解为什么是2
    p1Denom = 2.0  # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = codeNaiveBayes.loadDataSet()  # 创建实验样本
    myVocabList = codeNaiveBayes.createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(codeNaiveBayes.setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(codeNaiveBayes.setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(codeNaiveBayes.setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')


if __name__ == '__main__':
    postingList, classVec = codeNaiveBayes.loadDataSet()
    myVocabList = codeNaiveBayes.createVocabList(postingList)
    trainMat = []
    for postingDoc in postingList:
        trainMat.append(codeNaiveBayes.setOfWords2Vec(myVocabList, postingDoc))
    p0V, p1V, PAb = trainNB0(trainMat, classVec)
    print(p0V)
    print("p0V:%.2f" % 0.0)
    testingNB()
