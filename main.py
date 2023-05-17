from sklearn.feature_extraction.text import CountVectorizer
import pickle
import jieba
import joblib
from flask import Flask, request
import numpy as np

app = Flask(__name__)
count_stop_vec = CountVectorizer(vocabulary=pickle.load(open("CountVectorizer_features_light_gbm.pkl", 'rb')))
gbdt = joblib.load("light.m")


@app.route("/predict", methods=['POST'])
def predict():
    text = request.json.get("text").strip()
    texts = list(jieba.cut(text, cut_all=False))
    x_count_stop_dev = count_stop_vec.transform([' '.join(texts)])
    proba = gbdt.predict(x_count_stop_dev.toarray())
    index = np.argmax(proba[0])
    if index == 0:
        label = '会议通知'
    elif index == 1:
        label = '培训通知'
    else:
        label = '其他通知'
    return {'label': label}


if __name__ == '__main__':
    app.run(port=8030, debug=True, host='172.16.19.81')  # 启动服务
