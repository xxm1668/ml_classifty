import os
import jieba
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score


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


if __name__ == '__main__':
    folder_path = './Database/SogouC/Sample'
    x_train, x_test, y_train, y_test = TextProcessing(folder_path, test_size=0.2)

    count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
    x_count_stop_train = count_stop_vec.fit_transform(x_train).toarray()
    x_count_stop_test = count_stop_vec.transform(x_test)
    feature_path = 'CountVectorizer_features_light_gbm.pkl'  # 保存路径
    with open(feature_path, 'wb') as fw:
        pickle.dump(count_stop_vec.vocabulary_, fw)
    y_train2 = []
    for y in y_train:
        if y == 'meeting':
            y_train2.append(0)
        elif y == 'notice':
            y_train2.append(1)
        else:
            y_train2.append(2)
    y_test2 = []
    for y in y_test:
        if y == 'meeting':
            y_test2.append(0)
        elif y == 'notice':
            y_test2.append(1)
        else:
            y_test2.append(2)
    y_train = y_train2
    d_train = lgb.Dataset(x_count_stop_train, label=np.array(y_train))
    # setting up the parameters
    params = {}
    params['learning_rate'] = 0.03
    params['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
    params['objective'] = 'multiclass'  # Multi-class target feature
    params['metric'] = 'multi_logloss'  # metric for multi-class
    params['max_depth'] = 10
    params['num_class'] = 3
    # training the model
    clf = lgb.train(params, d_train, 100)  # training the model on 100 epocs
    # prediction on the test dataset
    y_pred_1 = clf.predict(x_count_stop_test.toarray())
    y_pred_2 = []
    for pred in y_pred_1:
        y_pred_2.append(np.argmax(pred))
    score = precision_score(y_pred_2, y_test2, average=None).mean()
    joblib.dump(clf, 'light.m')
    text = '小明明早8点半去304室参加培训'
    texts = list(jieba.cut(text, cut_all=False))
    x_count_stop_dev = count_stop_vec.transform([' '.join(texts)]).toarray()
    result = clf.predict(x_count_stop_dev)
    print(result)
    clf1 = joblib.load('light.m')
    print(clf1.predict(x_count_stop_dev))
