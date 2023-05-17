# coding: utf-8
import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set


def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 100:  # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file), 'r') as fp:
                raw = fp.read()
            # print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------
            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    ## 划分训练集和测试集
    train_data_list, test_data_list, train_class_list, test_class_list = train_test_split(data_list, class_list,
                                                                                          test_size=test_size,
                                                                                          shuffle=True)
    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    all_words_list = []
    for i in all_words_tuple_list:
        all_words_list.append(i[0])

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度1000
            break
        # print all_words_list[t]
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word: 1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk',
                   best_accary=0.0):
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # print classifier.predict(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0],
        # print ''
        test_accuracy = classifier.score(test_feature_list, test_class_list)
        print('精确度：', test_accuracy)
        if test_accuracy >= best_accary:
            joblib.dump(classifier, "clf_model.m")
            best_accary = test_accuracy
    else:
        test_accuracy = []
    return test_accuracy, best_accary


def test(test_features):
    clf = joblib.load("clf_model.m")
    ls = []
    proba = clf.predict_proba(test_features)
    for i in range(len(proba[0])):
        lk = str(i) + ":" + str(proba[0][i])
        ls.append(lk)
    return ls


if __name__ == '__main__':

    print("start")

    ## 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                        test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # flag = 'nltk'
    flag = 'sklearn'
    deleteNs = range(0, 1000, 20)
    test_accuracy_list = []
    best_accary = 0.0
    best_score = 0.0
    best_feature_words = []
    for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)
        test_accuracy, best_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list,
                                                      test_class_list, flag, best_accary)
        if best_accuracy > best_score:
            best_feature_words = feature_words
            best_score = best_accary
        test_accuracy_list.append(test_accuracy)
    print(test_accuracy_list)

    # 结果评价
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.savefig('result.png')

    print("finished")

    texts = '会议通知'
    word_cut = jieba.cut(texts, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
    word_list = list(word_cut)
    test_f, dev = TextFeatures([word_list], [], best_feature_words, flag)
    results = test(test_f)
    print(results)
