import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import SimpleKNN


def file2matrix(file_path):
    with open(file_path) as fr:
        array_lines = fr.readlines()  # 将结果按照行划分为数组
    array_lines_count = len(array_lines)

    return_mat = np.zeros((array_lines_count, 3))
    label_vector = []
    index = 0

    for line in array_lines:
        line = line.strip()  # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        if list_from_line[-1] == 'didntLike':
            label_vector.append(1)
        elif list_from_line[-1] == 'smallDoses':
            label_vector.append(2)
        elif list_from_line[-1] == 'largeDoses':
            label_vector.append(3)
        index += 1
    return return_mat, label_vector


def show_data(attributeMat, labelVec):
    """将得到的数据可视化显示
    :param attributeMat:特征矩阵
    :param labelVec:标签向量
    :return:
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    number_of_labels = len(labelVec)
    labels_colors = []
    for item in labelVec:
        if item == 1:
            labels_colors.append('black')
        if item == 2:
            labels_colors.append('orange')
        if item == 3:
            labels_colors.append('red')
    axs[0][0].scatter(x=attributeMat[:, 0], y=attributeMat[:, 1], color=labels_colors, s=15, alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=attributeMat[:, 0], y=attributeMat[:, 2], color=labels_colors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=attributeMat[:, 1], y=attributeMat[:, 2], color=labels_colors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    plt.show()


def auto_norm(data_set):
    """将数据进行归一化操作
    :param data_set:特征矩阵
    :return:
    """
    min_vals = data_set.min(axis=0)
    max_vals = data_set.max(axis=0)
    ranges = max_vals - min_vals
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]

    norm_data_set = data_set - np.tile(min_vals, (m, 1))  # 对最小值进行扩充
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    return norm_data_set, ranges, min_vals


def dating_test_set():
    file_name = "data.txt"
    attributeMat, labelVex = file2matrix(file_name)
    # show_data(attributeMat, labelVex)
    test_ratio = 0.10  # 测试集占比
    normMat, ranges, minVals = auto_norm(attributeMat)
    m = normMat.shape[0]
    num_of_test = int(m * test_ratio)
    error_count = 0.0

    for i in range(num_of_test):
        classifierResult = SimpleKNN.classify0(normMat[i, :], normMat[num_of_test:m, :],
                                               labelVex[num_of_test:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, labelVex[i]))
        if classifierResult != labelVex[i]:
            error_count += 1.0
    print("错误率:%f%%" % (error_count / float(num_of_test) * 100))


def classifyPerson():
    result_list = ['讨厌', '有些喜欢', '非常喜欢']
    percent_of_vedio_game = float(input("玩视频游戏所耗时间百分比："))
    miles_of_fly = float(input("每年获得的飞行常客里程数："))
    ice_cream = float(input("每周消费的冰淇淋公升数："))
    file_name = "data.txt"
    data_mat, data_labels = file2matrix(file_name)
    norm_mat, ranges, min_vals = auto_norm(data_mat)

    in_arr = np.array([miles_of_fly, percent_of_vedio_game, ice_cream])
    norm_in_arr = (in_arr - min_vals) / ranges
    classifier_result = SimpleKNN.classify0(norm_in_arr, norm_mat, data_labels, 3)
    print("你可能%s这个人" % (result_list[classifier_result - 1]))


if __name__ == '__main__':
    # dating_test_set()
    classifyPerson()
