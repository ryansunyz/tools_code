# -*- coding: UTF-8 encoding='cp1252' -*-

import numpy as np
import random
import re  # 正则表达式
import codeNaiveBayes
import LaplaceSmoothing


def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def creatVocabList(dataSet):
    """创建词汇表
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def bagOfWords2VecMN(vocablist, inputSet):
    """根据词袋 构造词向量
    :param vocablist:
    :param inputSet:
    :return:
    """
    returnVec = 0 * len(vocabList)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] += 1
            return returnVec


def spamTest():
    """测试朴素贝叶斯分类器
    :return:
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('./data/spam/%d.txt' % i, encoding='gbk', mode='r').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('./data/ham/%d.txt' % i, encoding='gbk', mode='r').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = creatVocabList(docList)
    trainingSet = list(range(50));
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(codeNaiveBayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = LaplaceSmoothing.trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = codeNaiveBayes.setOfWords2Vec(vocabList, docList[docIndex])
        if LaplaceSmoothing.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的数据集%d：" % docIndex, docList[docIndex])
    print('错误率:%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
