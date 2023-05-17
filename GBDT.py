import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool, Manager
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pickle


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


class GBDTClassifier:
    def __init__(self, X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.features = np.array(X)
        if len(self.features.shape) == 1:
            self.features = self.features.reshape(1, -1)
        self.labels = np.array(y).reshape(-1, 1)
        self.labels[self.labels == 0] = -1
        self.estimators = []
        self.tree_predict_value = []
        self.init_value = np.log((1 + np.mean(self.labels)) / (1 - np.mean(self.labels)))
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _cal_cj(self, j, r, area_num, i, dic):
        '''
        计算叶节点j的预测值
        :param j: 节点名
        :param r: 本轮样本残差（待预测值）
        :param area_num: 本轮样本所在节点
        :param i: 轮数
        :return: 无返回，将计算结果存入字典中
        '''
        # if tree_predict_value is None:
        #     tree_predict_value = self.tree_predict_value
        rj = r[area_num == j]
        c = np.sum(rj) / np.sum(abs(rj) * (1 - abs(rj)))
        dic[j] = c

    def fit(self):
        fk_1 = self.init_value * np.ones([self.labels.shape[0], 1])
        for i in tqdm(range(self.n_estimators)):
            # 每一轮迭代，首先计算残差r，然后训练回归树，并获取每个样本所在的叶节点
            dic = {}
            r = self.labels / (1 + np.exp(self.labels * fk_1))
            #             print('r',np.unique(r))
            dtr = DecisionTreeRegressor(random_state=1, max_depth=self.max_depth).fit(self.features, r)
            self.estimators.append(dtr)
            area_num = dtr.apply(self.features)
            # 并行计算每个节点的预测值
            for j in np.unique(area_num):
                self._cal_cj(j, r, area_num, i, dic)
            self.tree_predict_value.append(dic)
            ci = np.array(list(map(lambda x: self.tree_predict_value[i][x], area_num)))
            #             print('ci',i,ci[0],fk_1[0],self.labels[0])
            fk_1 = fk_1 + self.learning_rate * ci.reshape(-1, 1)

    def _predict(self, i, X):
        area_num = self.estimators[i].apply(X)
        return np.array(list(map(lambda x: self.learning_rate * self.tree_predict_value[i][x], area_num)))

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        with Pool() as p:
            result = np.array(list(p.starmap(self._predict, [(i, X) for i in range(self.n_estimators)])))
            result[-1, :] = result[-1, :] / self.learning_rate
            return np.array(np.sum(result, axis=0) >= 0).astype(int)


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)

    folder_path = './Database/SogouC/Sample'
    all_words_list, x_train, x_test, y_train, y_test = TextProcessing(folder_path, test_size=0.2)

    count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
    x_count_stop_train = count_stop_vec.fit_transform(x_train).toarray()
    x_count_stop_test = count_stop_vec.transform(x_test)
    feature_path = 'CountVectorizer_features_gbdt.pkl'  # 保存路径
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
    y_train = y_train2
    gbdt = GBDTClassifier(x_count_stop_train, y_train)
    gbdt.fit()
    print('训练完成')
    print(f"本例准确率：{accuracy_score(y_test2, gbdt.predict(x_count_stop_test))}")
    joblib.dump(gbdt, 'gbdt.m')
    text = '小明明早8点半去304室开会'
    texts = list(jieba.cut(text, cut_all=False))
    x_count_stop_dev = count_stop_vec.transform([' '.join(texts)]).toarray()
    result = gbdt.predict(x_count_stop_dev)
    print(result)

    gbdt2 = joblib.load("gbdt.m")
    result = gbdt2.predict(x_count_stop_dev)
    print(result, '----')

    gbdt_sklearn = GradientBoostingClassifier(criterion='mse').fit(x_count_stop_train, y_train)
    print(f"sklearn准确率：{accuracy_score(y_test2, gbdt_sklearn.predict(x_count_stop_test))}")
    print(gbdt_sklearn.score(x_count_stop_test, y_test2))
