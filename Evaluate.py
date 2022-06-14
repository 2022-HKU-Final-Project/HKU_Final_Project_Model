#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: zhangyuhao
@file: Evaluate.py
@time: 2022/6/6 下午10:56
@email: yuhaozhang76@gmail.com
@desc: 
"""
from simpletransformers.simpletransformers.model import TransformerModel
from simpletransformers.classification.classification_model import ClassificationModel
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
data = pd.read_csv("./data/train_set_6w.csv")


le = preprocessing.LabelEncoder()
le.fit(np.unique(data['true_job'].tolist()))
data['true_job'] = data['true_job'].apply(lambda x:le.transform([x])[0])
np_labels = len(data.jobPosition.value_counts())
print(np_labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(data['jobDesc'], data['jobPosition'], test_size=.2, random_state=0)

print('train performance:', end='\n')
model = ClassificationModel('distilbert', './distilbert_outputs/best_model/', num_labels=np_labels)
train_y, _ = model.predict(train_texts.tolist())
print(classification_report(le.inverse_transform(train_labels),le.inverse_transform(train_y)))

print('validation performance:', end='\n')
val_y, _ = model.predict(val_texts.tolist())
print(classification_report(le.inverse_transform(val_labels),le.inverse_transform(val_y)))
