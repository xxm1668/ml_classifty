import jieba
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pickle

count_stop_vec = CountVectorizer(vocabulary=pickle.load(open("CountVectorizer_features.pkl", 'rb')))
text = '请小明童鞋明早7点来302室开会'
texts = list(jieba.cut(text, cut_all=False))
x_count_stop_dev = count_stop_vec.transform([' '.join(texts)])
clf = joblib.load("clf_model.m")
ls = []
proba = clf.predict_proba(x_count_stop_dev)
print(proba)
