# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:22:39 2023

@author: 14896
"""

import sys
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import LinearSVR,SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split,GridSearchCV,LeaveOneOut
from sklearn.feature_selection import RFECV,RFE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from minepy import MINE
import joblib
#导入数据
data = np.loadtxt('data10000.txt', delimiter='\t')
data = np.random.permutation(data)
data_input = data[:,(0,3,4,5,6,7,8,12,13)] #输入参量
data_target = data[:,[-1]]
#归一化
data_input_normed_0=(data_input-data_input.min(axis=0))/(data_input.max(axis=0)-data_input.min(axis=0))
data_target_normed=(data_target-data_target.min(axis=0))/(data_target.max(axis=0)-data_target.min(axis=0))

# P = np.corrcoef(data_input_normed.T)
# SVR (C = i, epsilon= j)
for a in range(0,9,1):
    for b in range(a+1,9,1):
        for c in range(b+1,9,1):
           for d in range(c+1,9,1):
               for e in range(d+1,9,1):
                   data_input_normed =  data_input_normed_0[:,(a,b,c,d,e)]
                   results_errors = np.zeros((0,4), dtype=float)
                   for i in [1.0]:
                       for j in [0.06]:
                             num = 0
                             mean_num_errors_01 = np.zeros((0,2), dtype=float)
                             mean_num_errors_02 = np.zeros((0,2), dtype=float)
                             mean_num_errors_11 = np.zeros((0,2), dtype=float)
                             mean_num_errors_12 = np.zeros((0,2), dtype=float)
                             for num in range (0,10,1):
                                 num = num+1
                                 simulator = SVR(C = i, epsilon= j)
                                 predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                                 score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                                 predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化
                                 MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                                 MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                                 MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                                 RMSE = np.sqrt(MSE)
                                 error_01 = [[np.mean(score), MAE], ]
                                 error_02 = [[np.mean(score), MAPE], ]
                                 error_11 = [[np.mean(score), MSE], ]
                                 error_12 = [[np.mean(score), RMSE], ]
                                 mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                                 mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                                 mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                                 mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
                                 mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
                                 mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
                                 mean_score = np.mean(abs(mean_num_errors_02[:,0]))
                                 mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
                                 mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
                                 error_2 =  [[i, j,  mean_score, mean_MSE],] 
                                 results_errors = np.append(results_errors, error_2, axis=0)
                   print('输入参数：',a,',',b,',',c,',',d,',',e,'    R2:',mean_score, ' MAE:', mean_MAE,' MSE:',mean_MSE,' RMSE:',mean_RMSE,' MAPE:',mean_MAPE*100)
        