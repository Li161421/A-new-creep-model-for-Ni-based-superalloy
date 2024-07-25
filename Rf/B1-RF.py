import numpy as np
import pandas as pd           #读取数据
from sklearn.model_selection import train_test_split           #划分训练集与测试集
df = pd.read_csv("特征参数数据.csv",encoding="gbk")
#将数据分为训练和测试集
x = df.iloc[0:186913,0:13]#15和16列是目标性能，其余的14列均为参数
y = df.iloc[0:186913,14]          #输出12、13列为目标性能
feature_list = list(x.columns)        #输入名称
#x = np.array(x)            #格式转换
#归一化
x_normed=(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
x_normad = np.array(x_normed)
y_normed=(y-y.min(axis=0))/(y.max(axis=0)-y.min(axis=0))
y_normad = np.array(y_normed)
#划分训练集与测试集
train_x, test_x, train_y, test_y = train_test_split(x_normed, y_normed, test_size = 0.2,random_state =26)
#初步建立随机森林模型
from sklearn.ensemble import RandomForestRegressor #导入随机森林算法
from sklearn.metrics import r2_score              #用于模型拟合优度评估R2
from sklearn.metrics import mean_absolute_error   #MAE
from sklearn.metrics import mean_squared_error    #MSE
# 详细参考https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
rf = RandomForestRegressor()
rf = rf.fit(train_x, train_y)
score = rf.score(test_x, test_y)
print('默认参数下测试集评分：',score)
predictions= rf.predict(test_x)
print("train r2:%.3f"%r2_score(train_y,rf.predict(train_x)),"mae:%.3f"%mean_absolute_error(train_y, rf.predict(train_x)),"mse:%.5f"%mean_squared_error(train_y, rf.predict(train_x)))
print("test r2:%.3f"%r2_score(test_y,predictions),"mae:%.3f"%mean_absolute_error(test_y, predictions),"mse:%.5f"%mean_squared_error(test_y, predictions))

#进行贝叶斯优化调参
#由于每次运行RF模型的随机分配样本，故优化参数应该对应的是当下训练集和测试集的最优，最终得到的特征重要性可能值不一样
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestRegressor(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),  # float
                               max_depth=int(max_depth),
                               random_state=2),
        train_x, train_y, scoring='r2', cv=10).mean()
    return val
pbounds = {'n_estimators': (10, 250),  # 表示取值范围为10至250
           'min_samples_split': (2, 25),
           'max_features': (0.1, 0.999),
           'max_depth': (5, 50)}
optimizer = BayesianOptimization(
    f=rf_cv,  # 黑盒目标函数
    pbounds=pbounds,  # 取值空间
    verbose=2,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
    random_state=1)
optimizer.maximize(  # 运行
    init_points=5,  # 随机搜索的步数
    n_iter=50)  # 执行贝叶斯优化迭代次数
print(optimizer.max)  # 最好的结果与对应的参数

#使用max得到最优
# {'target': 0.9436266964413887, 'params': {'max_depth': 31.109560347828253, 'max_features': 0.36714342264348354, 'min_samples_split': 2.1945963586639192, 'n_estimators': 156.92237125868954}}

#本代码不可一步到位
#将搜索到的最优结果(四舍五入一下)带入回到元模型中
rf = RandomForestRegressor(max_depth = 12, max_features = 0.1,
                               min_samples_leaf = 3,  n_estimators = 172)
rf = rf.fit(train_x, train_y)
score = rf.score(test_x, test_y)
print('优化参数下测试集评分：',score)
predictions= rf.predict(test_x)
print("train r2:%.7f"%r2_score(train_y,rf.predict(train_x)),"mae:%.3f"%mean_absolute_error(train_y, rf.predict(train_x)),"mse:%.3f"%mean_squared_error(train_y, rf.predict(train_x)))
print("test r2:%.7f"%r2_score(test_y,predictions),"mae:%.3f"%mean_absolute_error(test_y, predictions),"mse:%.3f"%mean_squared_error(test_y, predictions))
importances = list(rf.feature_importances_)      #目标性能的重要性
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list,importances)]     #将相关变量名称与重要性对应
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)                #排序
[print('Variable: {:} Importance: {}'.format(*pair)) for pair in feature_importances]            #输出特征影响程度详细数据
#反归一化
