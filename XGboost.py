# 导入所需要的包
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
import os
import jieba
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib


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

    return train_data_list, test_data_list, train_class_list, test_class_list


folder_path = './Database/SogouC/Sample'
x_train, x_test, y_train, y_test = TextProcessing(folder_path, test_size=0.2)

count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
x_count_stop_train = count_stop_vec.fit_transform(x_train).toarray()
x_count_stop_test = count_stop_vec.transform(x_test)
feature_path = 'CountVectorizer_features_xgboost.pkl'  # 保存路径
with open(feature_path, 'wb') as fw:
    pickle.dump(count_stop_vec.vocabulary_, fw)
y_train2 = []
for y in y_train:
    if y == 'meeting':
        y_train2.append(0)
    else:
        y_train2.append(1)
y_test2 = []
for y in y_test:
    if y == 'meeting':
        y_test2.append(0)
    else:
        y_test2.append(1)
y_test = np.array(y_test2)
y_train = y_train2

model = xgb.XGBClassifier()
# 训练模型
model.fit(x_count_stop_train, np.array(y_train))
# 预测值
y_pred = model.predict(x_count_stop_test)

joblib.dump(model, 'xgboost.m')
'''
评估指标
'''
# 求出预测和真实一样的数目
true = np.sum(y_pred == y_test)
print('预测对的结果数目为：', true)
print('预测错的的结果数目为：', y_test.shape[0] - true)
# 评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

print('预测数据的准确率为： {:.4}%'.format(accuracy_score(y_test, y_pred) * 100))
print('预测数据的精确率为：{:.4}%'.format(
    precision_score(y_test, y_pred) * 100))
print('预测数据的召回率为：{:.4}%'.format(
    recall_score(y_test, y_pred) * 100))
# print("训练数据的F1值为：", f1score_train)
print('预测数据的F1值为：',
      f1_score(y_test, y_pred))
print('预测数据的Cohen’s Kappa系数为：',
      cohen_kappa_score(y_test, y_pred))
# 打印分类报告
print('预测数据的分类报告为：', '', classification_report(y_test, y_pred))
