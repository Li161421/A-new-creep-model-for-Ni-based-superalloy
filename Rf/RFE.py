# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:35:25 2023

@author: 14896
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
data=pd.read_csv('特征参数10000.csv')
# 数据清洗,删除无用特征，axis=1是按照列操作。
data.drop(['Ticket', 'PassengerId'], axis=1, inplace=True)
# 分类变量转换，通过Dataframe的repalce结合字典映射。
gender_mapper = {'male': 0, 'female': 1}
data['Sex'].replace(gender_mapper, inplace=True)
#提取Title字段里的提取出称谓并做二值化转换。
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].strip().split(' ')[0])
data['Title'] = [0 if x in ['Mr.', 'Miss.', 'Mrs.'] else 1 for x in data['Title']]
data = data.rename(columns={'Title': 'Title_Unusual'})
data.drop('Name', axis=1, inplace=True)
 
data['Cabin_Known'] = [0 if str(x) == 'nan' else 1 for x in data['Cabin']]
data.drop('Cabin', axis=1, inplace=True)
 
#对Embarked字段进行One-Hot编码，生成哑变量
emb_dummies = pd.get_dummies(data['Embarked'], drop_first=True, prefix='Embarked')
data = pd.concat([data, emb_dummies], axis=1)
data.drop('Embarked', axis=1, inplace=True)
# 用均值填充age字段
data['Age'] = data['Age'].fillna(int(data['Age'].mean()))
 
# 删除相关性高的特征，当前数据集特征无强相关性，所以correlated_features集合为空。
correlated_features = set()
correlation_matrix = data.drop('Survived', axis=1).corr()
 
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
data.drop(correlated_features)
 
#定义X(所有特征)和Target、y(目标变量)
X = data.drop('Survived', axis=1)
target = data['Survived']
# random_state=101，随机种子，为了数据的可再现。
rfc = RandomForestClassifier(random_state=101)
'''
estimator:某个模型实例,这里用的是随机森林
step:每次迭代时要删除的特征个数
cv:交叉验证,用StratifiedKFold并指定K是10
scoring:指定优化时的度量方法，这里选择是'accuracy',精确度。
'''
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)
#
print('优化后的特征个数是: {}'.format(rfecv.n_features_))
 
#调用grid_scores_画图。
plt.figure(figsize=(16, 9))
plt.title('RFE_交叉验证', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('被选择的特征', fontsize=14, labelpad=20)
plt.ylabel('% 选择的分类数', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.show()