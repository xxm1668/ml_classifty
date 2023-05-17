from sklearn.feature_extraction.text import CountVectorizer
import pickle
import jieba
import joblib
import numpy as np

count_stop_vec = CountVectorizer(vocabulary=pickle.load(open("CountVectorizer_features_light_gbm.pkl", 'rb')))
gbdt = joblib.load("light.m")


def predict(text):
    texts = list(jieba.cut(text, cut_all=False))
    x_count_stop_dev = count_stop_vec.transform([' '.join(texts)])
    proba = gbdt.predict(x_count_stop_dev.toarray())
    return proba[0]


if __name__ == '__main__':
    text = '请小明明天于8点半到405室参加培训'
    while 1:
        text = input('请输入：')
        result = predict(text)
        index = np.argmax(result)
        if index == 0:
            label = '会议通知'
        else:
            label = '培训通知'
        print(label)
