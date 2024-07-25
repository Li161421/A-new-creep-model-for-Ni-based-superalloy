# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:06:21 2023

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
data_input = data[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13)] #输入参量
data_target = data[:,[-1]]
#归一化
data_input_normed=(data_input-data_input.min(axis=0))/(data_input.max(axis=0)-data_input.min(axis=0))
data_target_normed=(data_target-data_target.min(axis=0))/(data_target.max(axis=0)-data_target.min(axis=0))

P = np.corrcoef(data_input_normed.T)
# SVR (C = i, epsilon= j)
results_errors = np.zeros((0,4), dtype=float)
for i in [1.0]:  
    for j in [0.06]:    
            num = 0
            mean_num_errors = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator = SVR(C = i, epsilon= j)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                
                error_1 = [[np.mean(score), MAE], ]
                mean_num_errors = np.append(mean_num_errors, error_1, axis=0)    
            mean_score = np.mean(abs( mean_num_errors[:,0]))
            mean_MAE = np.mean(abs(mean_num_errors[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MAE],] 
            results_errors = np.append(results_errors, error_2, axis=0)    
np.savetxt('results_errors-SVR.txt', results_errors)