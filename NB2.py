from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import os
import jieba
import joblib
from sklearn import svm
import numpy as np
import pickle

'''
文本特征提取：
    将文本数据转化成特征向量的过程
    比较常用的文本特征表示法为词袋法
词袋法：
    不考虑词语出现的顺序，每个出现过的词汇单独作为一列特征
    这些不重复的特征词汇集合为词表
    每一个文本都可以在很长的词表上统计出一个很多列的特征向量
    如果每个文本都出现的词汇，一般被标记为 停用词 不计入特征向量

主要有两个api来实现 CountVectorizer 和 TfidfVectorizer
CountVectorizer：
    只考虑词汇在文本中出现的频率
TfidfVectorizer：
    除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量
    能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征

相比之下，文本条目越多，Tfid的效果会越显著


下面对两种提取特征的方法，分别设置停用词和不停用，
使用朴素贝叶斯进行分类预测，比较评估效果

'''


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
            data_list.append(' '.join(word_list))
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


# 1 下载新闻数据
news = fetch_20newsgroups(subset="all")

# 2 分割训练数据和测试数据
## 文本预处理
folder_path = './Database/SogouC/Sample'
all_words_list, x_train, x_test, y_train, y_test = TextProcessing(folder_path, test_size=0.2)
# 3.1 采用普通统计CountVectorizer提取特征向量
# 去除停用词
count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
x_count_stop_train = count_stop_vec.fit_transform(x_train)
print(count_stop_vec.get_feature_names())
x_count_stop_test = count_stop_vec.transform(x_test)
text = '小明明早8点半去304室开会'
texts = list(jieba.cut(text, cut_all=False))
x_count_stop_dev = count_stop_vec.transform([' '.join(texts)])

feature_path = 'CountVectorizer_features.pkl'  # 保存路径
with open(feature_path, 'wb') as fw:
    pickle.dump(count_stop_vec.vocabulary_, fw)

# 3.2 采用TfidfVectorizer提取文本特征向量
# 去除停用词
tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
x_tfid_stop_train = tfid_stop_vec.fit_transform(x_train)
print(tfid_stop_vec.get_feature_names())
x_tfid_stop_test = tfid_stop_vec.transform(x_test)

# 4 使用朴素贝叶斯分类器  分别对两种提取出来的特征值进行学习和预测
# 对普通通统计CountVectorizer提取特征向量 学习和预测
# 去除停用词
mnb_count_stop = MultinomialNB()
mnb_count_stop.fit(x_count_stop_train, y_train)  # 学习
mnb_count_stop_y_predict = mnb_count_stop.predict(x_count_stop_test)  # 预测
joblib.dump(mnb_count_stop, "clf_model.m")
result = mnb_count_stop.predict(x_count_stop_dev)
print('---')

# 对TfidfVectorizer提取文本特征向量 学习和预测
# 去除停用词
mnb_tfid_stop = MultinomialNB()
mnb_tfid_stop.fit(x_tfid_stop_train, y_train)  # 学习
mnb_tfid_stop_y_predict = mnb_tfid_stop.predict(x_tfid_stop_test)  # 预测

# 5 模型评估
# 对普通统计CountVectorizer提取的特征学习模型进行评估
print("去除停用词的CountVectorizer提取的特征学习模型准确率：", mnb_count_stop.score(x_count_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_count_stop_y_predict, y_test))

# 对TfidVectorizer提取的特征学习模型进行评估
print("去除停用词的TfidVectorizer提取的特征学习模型准确率：", mnb_tfid_stop.score(x_tfid_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_tfid_stop_y_predict, y_test))

# 朴素贝叶斯预测
clf = joblib.load("clf_model.m")
ls = []
proba = clf.predict_proba(x_count_stop_dev)
print(proba)

clf = svm.SVC(C=10, gamma=0.8, max_iter=200)
clf.fit(x_count_stop_train, y_train)
train_result = clf.predict(x_count_stop_train)
precision = np.sum(train_result == y_train) / len(y_train)
print('Training precision:', precision)

test_result = clf.predict(x_count_stop_test)
precision = np.sum(test_result == y_test) / len(y_test)
print('Test precision:', precision)

test_result = clf.predict(x_count_stop_dev)
print(test_result)
