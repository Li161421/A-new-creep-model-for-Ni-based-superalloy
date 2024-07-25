% 导入数据
 load('data10000.mat')
 data=table2array(data_10000);
X = data(:, 1:end-1);
y = data(:, end);

% 定义超参数搜索范围
param_range = [0.1, 1, 10];
epsilon_range = [0.01, 0.1, 1];
gamma_range = [0.1, 1, 10];

% 定义十折交叉验证对象
cv = cvpartition(size(X,1), 'KFold', 10);

% 定义评价指标
mse_func = @(y, y_pred) mean((y - y_pred).^2);

% 使用十折交叉验证训练模型
mse_cv = zeros(cv.NumTestSets, 1);
for i = 1:cv.NumTestSets
    % 获取训练集和测试集
    train_idx = cv.training(i);
    test_idx = cv.test(i);
    X_train = X(train_idx,:);
    y_train = y(train_idx,:);
    X_test = X(test_idx,:);
    y_test =y(test_idx,:);

% 定义超参数搜索空间
param_space = hyperparameters('fitrsvm', X_train, y_train, 'OptimizeHyperparameters','all', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus'), ...
    'KernelFunction',optimizableVariable('RBF',{'auto'},'Type','categorical'), ...
    'BoxConstraint',optimizableVariable('C',param_range,'Type','real','Transform','log'), ...
    'Epsilon',optimizableVariable('epsilon',epsilon_range,'Type','real'), ...
    'KernelScale',optimizableVariable('gamma',gamma_range,'Type','real','Transform','log'));

end
% 使用超参数搜索空间训练模型
    mdl = fitrsvm(X_train, y_train, 'KernelFunction', 'RBF', 'HyperparameterOptimizationOptions', param_space);

% 在测试集上评估模型性能
    y_pred = predict(mdl, X_test);
    mse_cv(i) = mse_func(y_test, y_pred);

% % 输出交叉验证的平均误差和标准差
mse_mean = mean(mse_cv);
mse_std = std(mse_cv);
fprintf('Average MSE: %f\n', mse_mean);
fprintf('MSE standard deviation: %f\n', mse_std);