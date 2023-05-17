# 导入所需要的包
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import os
import jieba
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import logging

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
feature_path = 'CountVectorizer_features_xgboost2.pkl'  # 保存路径
with open(feature_path, 'wb') as fw:
    pickle.dump(count_stop_vec.vocabulary_, fw)
y_train2 = []
for y in y_train:
    if y == 'meeting':
        y_train2.append(0)
    else:
        y_train2.append(1)
y_train = y_train2

# 参数
parameters = {
    'max_depth': [5, 10, 15, 20, 25],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'n_estimators': [500, 1000, 2000, 3000, 5000],
    'min_child_weight': [0, 2, 5, 10, 20],
    'max_delta_step': [0, 0.2, 0.6, 1, 2],
    'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

}

xlf = xgb.XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=2000,
                        silent=True,
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=0)
# 训练模型
gs = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
gs.fit(x_count_stop_train, np.array(y_train))
print("Best score: %0.3f" % gs.best_score_)
logging.debug("Best score: %0.3f" % gs.best_score_)
print("Best parameters set: %s" % gs.best_params_)
logging.debug("Best parameters set: %s" % gs.best_params_)
