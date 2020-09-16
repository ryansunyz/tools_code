import os
import jieba
import random
from sklearn.naive_bayes import MultinomialNB


def TextProcessing(folder_path, test_size=0.2):
    """数据预处理
    :param folder_path:
    :return:
    """
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw, cut_all=False)
            word_list = list(word_cut)
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list
    print(data_list)
    print(class_list)


def MakeWordsSet(stop_words_file):
    """读取停用词
    :param stop_words_file:
    :return:
    """
    stop_words_set = set()
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                stop_words_set.add(word)
    return stop_words_set


def words_dict(all_words_list, deleteN, stop_words_set=set()):
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stop_words_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


import matplotlib.pyplot as plt

if __name__ == '__main__':
    folder_path = './data/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path)
    stop_words_file = './stopwords_cn.txt'
    stop_words_set = MakeWordsSet(stop_words_file)
    feature_words = words_dict(all_words_list, 100, stop_words_set)
    print(feature_words)
    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stop_words_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)

    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
