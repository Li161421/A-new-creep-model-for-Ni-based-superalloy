clc;clear;
load data.mat
% % 对应高温屈服强度数据5.6.xlsx
%% 1.读取数据
rowrank = randperm(size(data, 1)); % size获得table_data的行数，randperm打乱各行的顺序
table_new = data(rowrank,:); %生成打乱后的新表
x = table_new(:,3:12)';
t = table_new(:,2)';   

%% 构建前馈神经网络
% Choose a Training Function 详情见https://ww2.mathworks.cn/help/deeplearning/ug/train-and-apply-multilayer-neural-networks.html
% % 采用拟牛顿/贝叶斯正则化函数提高网络泛化能力
trainFcn = 'trainbr';  
hiddenLayerSize = 15;
net = feedforwardnet(hiddenLayerSize,trainFcn);
% 设置训练参数
net.trainParam.epochs = 2000; % 最大训练轮次
% 划分数据集(训练/验证/测试)
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% 训练100个神经网络取平均
numNN = 10;
nets = cell(1,numNN);
for i = 1:numNN
    fprintf('Training %d/%d\n', i, numNN);
    nets{i} = train(net, x, t);
end
perfs = zeros(1, numNN);
yTotal = 0;
for i = 1:numNN
    neti = nets{i};
    y = neti(x);
    perfs(i) = mse(neti, t, y);
    yTotal = yTotal + y;
end
perfs;
y_Average_Output = yTotal / numNN;
MSE = mse(nets{1}, t, y_Average_Output)
MAE = mae(nets{1}, t, y_Average_Output)
disp(['均方误差MAE：',num2str(MAE)])
disp(['均方误差MSE：',num2str(MSE)])
% disp(['相关系数R2： ',num2str(R1^2)])