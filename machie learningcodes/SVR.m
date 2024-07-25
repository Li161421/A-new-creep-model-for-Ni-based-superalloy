% 导入数据
 load('data10000.mat')
 data=table2array(data_10000);
X = data(:, 1:end-2);
y = data(:, end);

% 数据标准化处理
X_norm = zscore(X);
y_norm = zscore(y);

% 定义SVR模型
svr = fitrsvm(X_norm, y_norm, 'KernelFunction', 'rbf', 'KernelScale', 1, 'BoxConstraint', 10, 'Epsilon', 0.01);

% 进行预测
y_pred_norm = predict(svr, X_norm);
y_pred = y_pred_norm .* std(y) + mean(y); % 反标准化

% 计算模型评估指标
MAE = mean(abs(y_pred - y));
RMSE = sqrt(mean((y_pred - y).^2));
R2 = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);

% 输出结果
fprintf('MAE: %.4f\n', MAE);
fprintf('RMSE: %.4f\n', RMSE);
fprintf('R2: %.6f\n', R2);