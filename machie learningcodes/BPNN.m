%% 原始代码摘自https://blog.csdn.net/baoliang12345/article/details/128065676
clc;clear;
%% 1.读取数据
 load('data10000.mat')
data = table2array(data_10000);
rowrank = randperm(size(data, 1)); % size获得table_data的行数，randperm打乱各行的顺序
table_new =data(rowrank,:); %生成打乱后的新表
x = table_new(:,1:13);  %通过随机森林筛选出来的特征,去除第4列
t = table_new(:,15);       %15列为目标性能
% % 对应高温屈服强度数据5.6.xlsx

%% 2.设置训练集和测试集
trainNum = fix(size(data, 1)*0.8);
x_train = x(1:trainNum,:)'; 
% 训练集输入
t_train =t(1:trainNum)';                    % 训练集输出
x_test =x(trainNum+1:end,:)';    % 测试集输入
t_test =t(trainNum+1:end)';    % 测试集输出

%% 3.数据归一化
[xn_train,xps]=mapminmax(x_train,0,1);         % 训练集输入归一化到[0,1]之间
[tn_train,tps]=mapminmax(t_train,0,1);          % 训练集输出归一化到默认区间[0, 1]
xn_test=mapminmax('apply',x_test,xps);   % 测试集输入采用和训练集输入相同的归一化方式

%% 4.求解最佳隐含层
xnum=size(x,2);   %size用来求取矩阵的行数和列数，1代表行数，2代表列数
tnum=size(t,2);
disp(['输入层节点数：',num2str(xnum),',  输出层节点数：',num2str(tnum)])
disp(['隐含层节点数范围为 ',num2str(fix(sqrt(xnum+tnum))+1),' 至 ',num2str(fix(sqrt(xnum+tnum))+10)])
disp(' ')
disp('最佳隐含层节点的确定...')
 
%根据hiddennum=sqrt(m+n)+a，m为输入层节点数，n为输出层节点数，a取值[1,10]之间的整数
MSE=1e+5;                             %误差初始化
transform_func={'tansig','purelin'};  %激活函数采用tan-sigmoid和purelin
train_func='trainbr';                 %训练算法
for hiddennum=fix(sqrt(xnum+tnum))+1:fix(sqrt(xnum+tnum))+10
    net=newff(xn_train,tn_train,hiddennum,transform_func,train_func); %构建BP网络
    % 设置网络参数
    net.trainParam.epochs=3000;       % 设置训练次数
    net.trainParam.lr=0.01;           % 设置学习速率
    net.trainParam.goal=0.000001;     % 设置训练目标最小误差
    % 进行网络训练
    net=train(net,xn_train,tn_train);
    an0=sim(net,xn_train);     %仿真结果
    mse0=mse(tn_train,an0);   %仿真的均方误差
    disp(['当隐含层节点数为',num2str(hiddennum),'时，训练集均方误差MSE为：',num2str(mse0)])   
    %不断更新最佳隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳隐含层节点数为：',num2str(hiddennum_best),'，均方误差MSE为：',num2str(MSE)])

%% 5.构建最佳隐含层的BP神经网络
net=newff(xn_train,tn_train,hiddennum_best,transform_func,train_func);

% 网络参数
net.trainParam.epochs=3000;         % 训练次数
net.trainParam.lr=0.01;             % 学习速率
net.trainParam.goal=0.000001;       % 训练目标最小误差

%% 6.网络训练
net=train(net,xn_train,tn_train);      % train函数用于训练神经网络，调用蓝色仿真界面

%% 7.网络测试
an=sim(net,xn_test);                     % 训练完成的模型进行仿真测试
test_simu=mapminmax('reverse',an,tps);  % 测试结果反归一化
error=test_simu-t_test;                 % 测试值和真实值的误差

% 权值阈值
W1 = net.iw{1, 1};  %输入层到中间层的权值
B1 = net.b{1};      %中间各层神经元阈值
W2 = net.lw{2,1};   %中间层到输出层的权值
B2 = net.b{2};      %输出层各神经元阈值
% 
% % 8.结果输出
% % BP预测值和实际值的对比图
% figure
% plot(t_test,'bo-','linewidth',1.5)
% hold on
% plot(test_simu,'rs-','linewidth',1.5)
% legend('实际值','预测值')
% xlabel('测试样本'),ylabel('指标值')
% title('BP预测值和实际值的对比')
% set(gca,'fontsize',12)
% 
% BP测试集的预测误差图
% figure
% plot(error,'bo-','linewidth',1.5)
% xlabel('测试样本'),ylabel('预测误差')
% title('BP神经网络测试集的预测误差')
% set(gca,'fontsize',12)

%计算各项误差参数
[~,len]=size(t_test);            % len获取测试样本个数，数值等于testNum，用于求各指标平均值
SSE1=sum(error.^2);                   % 误差平方和
MAE1=sum(abs(error))/len;             % 平均绝对误差
MSE1=error*error'/len;                % 均方误差
RMSE1=MSE1^(1/2);                     % 均方根误差
MAPE1=mean(abs(error./t_test));  % 平均百分比误差
r=corrcoef(t_test,test_simu);    % corrcoef计算相关系数矩阵，包括自相关和互相关系数
R1=r(1,2);    
% 显示各指标结果
disp(' ')
disp('各项误差指标结果：')
disp(['误差平方和SSE：',num2str(SSE1)])
disp(['平均绝对误差MAE：',num2str(MAE1)])
disp(['均方误差MSE：',num2str(MSE1)])
disp(['均方根误差RMSE：',num2str(RMSE1)])
disp(['平均百分比误差MAPE：',num2str(MAPE1*100),'%'])
disp(['预测准确率为：',num2str(100-MAPE1*100),'%'])
disp(['相关系数R2： ',num2str(R1^2)])
% % 绘制热力图
% heatmap(R1^2);
% % 添加颜色条
% % colorbar;
% % % 添加标题和标签
% % title('相关系数矩阵');
% % xlabel('去除G参数');
% % ylabel('蠕变寿命');
% % 调整字体大小
% ax = gca;
% ax.FontSize = 12;
%显示测试集结果
% disp(' ');
% disp('测试集结果：');
% disp('    编号     实际值     BP预测值     误差');
% for i=1:len
%     disp([i,output_test(i),test_simu(i),error(i)]);   % 显示顺序: 样本编号，实际值，预测值，误差
% end