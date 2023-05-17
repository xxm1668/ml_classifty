import jieba
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.tree import DecisionTreeRegressor


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


count_stop_vec = CountVectorizer(vocabulary=pickle.load(open("CountVectorizer_features_gbdt.pkl", 'rb')))
text = '请小明童鞋明早7点来302室参加培训'
texts = list(jieba.cut(text, cut_all=False))
x_count_stop_dev = count_stop_vec.transform([' '.join(texts)])
gbdt = joblib.load("gbdt.m")
ls = []
proba = gbdt.predict(x_count_stop_dev)
print(proba)
