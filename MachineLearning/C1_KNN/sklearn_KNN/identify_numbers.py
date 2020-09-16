import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN


def img2vector(filename):
    """将32*32的二进制图像变为1*1024的向量
    :param filename:
    :return: 二进制图像的1*1024向量
    """
    fr = open(filename)
    lineStrList = fr.readlines()
    vec = []
    for line in lineStrList:
        line = line.strip()
        for i in range(32):
            vec.append(int(line[i]))
    vec = np.asarray(vec)
    return vec


def handwritingClassify():
    """手写数字识别
    :return:
    """
    hwLabels = []  # 测试集标签
    trainingFileList = listdir('trainingDigits')
    number_of_files = len(trainingFileList)
    trainingMat = np.zeros((number_of_files, 1024))
    for i in range(number_of_files):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        # classNumber=int(fileNameStr[0])
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    neigh = KNN(n_neighbors=3, algorithm='auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    number_of_files_test = len(testFileList)
    for i in range(number_of_files_test):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vector_of_test = img2vector('testDigits/%s' % (fileNameStr))
        vector_of_test = np.asarray(vector_of_test)
        vector_of_test = np.expand_dims(vector_of_test, axis=0)
        classfier_result = neigh.predict(vector_of_test)
        print("分类返回结果为%d\t真实结果为%d" % (classfier_result, classNumber))
        if (classfier_result != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / number_of_files_test * 100))


def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    # 构建kNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        vectorUnderTest = np.asarray(vectorUnderTest)
        vectorUnderTest = np.expand_dims(vectorUnderTest, 0)  # unsqueeze
        classifierResult = neigh.predict(vectorUnderTest)
        # print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
            print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
            print(fileNameStr)
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassify()
    # handwritingClassTest()
