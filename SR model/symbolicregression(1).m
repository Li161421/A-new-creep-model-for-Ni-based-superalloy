% 实验1：x1-x9(去除x7)->y
% 实验2：引入新特征x10 x11 x12
% 实验3：x1-x9(去除x7)->x7
% 实验4：引入领域变量x13 x14 x15

%lwj需要的结果 mse_test_all pop_all r2
clear
clc
dbstop if error
cd(fileparts(mfilename('fullpath')));addpath(genpath(cd));
load data0717.mat

% xdata(:,7) = zeros(size(xdata,1),1);
% xdata(:,8) = normalize(xdata(:,8));
data = data(1:37532,:);
xdata = data(:,1:9);
xdata(:,7) = zeros(size(xdata,1),1);
xdata(:,10) = xdata(:,1).^2./xdata(:,3);%符号回归提取出来的因子
xdata(:,11) = xdata(:,1).*xdata(:,5)./xdata(:,4);%符号回归提取出来的因子
xdata(:,12) = xdata(:,1).*xdata(:,2).^2./xdata(:,4);%符号回归提取出来的因子
% ydata = data(:,end);



xdata(:,13) = xdata(:,2).*xdata(:,9).*xdata(:,8);
xdata(:,14) = xdata(:,4)./xdata(:,2)./xdata(:,9);
xdata(:,15) = (ones(size(xdata,1),1)./xdata(:,2)).^5;
xdata(:,16) = xdata(:,4).*xdata(:,5);
ydata = data(:,10);



idx_f = [];
pop_all = [];
count_x_all = [];
mse_train_all = [];
mse_test_all = [];

for time = 1:10
    knum = 1;
    opts = statset('Display','final','MaxIter',1000);
    [idx,~,sumd] = kmeans([xdata,ydata], knum,'Options',opts);

    for i = 1:knum
        a{i,:} = find(idx==i);
    end

    for j = 1:knum
        algo = 'Copy_of_sr_char_single_task_v1';
        gen = 1000;

        N = size(a{j,:},1);
        N_train = randperm(N,round(N*0.7));
        N_test = setdiff((1:N),N_train);
        dim = size(xdata,2);

        [x_train, y_train] = deal(xdata(a{j,:}(N_train), :), ydata(a{j,:}(N_train), :));
        [x_test, y_test] = deal(xdata(a{j,:}(N_test), :), ydata(a{j,:}(N_test), :));


        if contains(algo,{'Copy_of_sr_char_single_task_v1','sr_char_single_task_Frequent_v1'})
            [pop,count_x,mse_train,mse_test] = feval(algo, gen, x_train, y_train, x_test, y_test);

            for i = 1:floor(size(xdata,2)/10)
                if i < floor(size(xdata,2)/10)
                    count_x(i) = count_x(i)-sum(count_x(10*i:10*i+9));
                else
                    count_x(i) = count_x(i)-sum(count_x(10*i:end));
                end
            end
            count_x_all = [count_x_all;count_x];
            mse_train_all = [mse_train_all,mse_train];
            mse_test_all = [mse_test_all,mse_test];
            pop_all = [pop_all;pop];
        else
            [mse_train, mse_test, pop, complexity] = feval(algo, gen, x_train, y_train, x_test, y_test);
        end
    end

%     for i = 1:size(pop_all,1)
%         flag = 0;
%         bns = [];
%         cns = {};
%         try
%             ans = char(vpa(str2sym(pop_all(i,:))));
%         catch
%             continue;
%         end
%         for j = 1:size(ans,2)
%             if ans(1,j) == '('
%                 flag = 1;
%             elseif ans(1,j) == ')'
%                 flag = 0;
%             elseif ans(1,j) == '+' && flag == 0
%                 bns = [bns,j];
%             elseif ans(1,j) == '-' && flag == 0
%                 bns = [bns,j];
%             end
%         end
%         bns = [0,bns,size(ans,2)+1];
%         for j = 1:size(bns,2)-1
%             bbns{i+(time-1)*100,j} = ans(1,bns(j)+1:bns(j+1)-1);
%         end
%     end

    temp_all = [];
    temp_index_all = [];
    for i = 1:size(pop_all,1)
        disp(i);
        temp1 = pop_all{i,:};
        temp1 = strrep(temp1,'*','.*');
        temp1 = strrep(temp1,'/','./');
        temp1 = strrep(temp1,'^','.^');
        %     for j = 10:size(xdata,2)
        %         temp1 = strrep(temp1,['X',num2str(j)], ['xdata(:,',num2str(j),')']);
        %     end
        %     for j = 1:9
        %         temp1 = strrep(temp1,['X',num2str(j)], ['xdata(:,',num2str(j),')']);
        %     end
        for j = size(xdata,2):-1:1
            temp1 = strrep(temp1,['X',num2str(j)], ['xdata(:,',num2str(j),')']);
        end
        temp = eval(temp1);
        temp_abs = abs(temp-ydata);
        [temp_abs,index] = sort(temp_abs);
        temp_index_all = index(1:round(sqrt(size(xdata,1))),:);
        xxdata = xdata;


        for j = 1:size(xdata,2)
            xdata = xxdata;
            xdata(:,j) = xdata(:,j)*1.01;
            temp2(:,j) = eval(temp1);
            temp2_1(:,j) = abs(abs(temp2(:,j) - temp)./(xdata(:,j)*0.01));
        end


        temp2_2 = temp2_1(temp_index_all,:);
        temp2_all(i,:) = mean(temp2_2,1);
        temp_all = [temp_all,temp_abs];


    end

    final_1 = count_x;
    for i = 1:size(temp2_all,1)
        for j = 1:size(xdata,2)
            if isnan(temp2_all(i,j))
                temp2_all(i,j) = 0;
            end
        end
    end

    for i = 1:size(xdata,2)
        cc = temp2_all(:,i);
        ccc = find(cc);
        final_2(1,i) = mean(rmoutliers(cc(ccc)));
    end
    for i = 1:size(xdata,2)
        if isnan(final_2(i))
            final_2(i) = 0;
        end
        if isnan(final_1(i))
            final_1(i) = 0;
        end
    end
    final_3 = zeros(1,size(xdata,2));
    for i = 1:size(xdata,2)
        if final_1(i)<mean(final_1)
            final_3(i) = final_3(i) + size(final_3,2);
        end
        for j = i+1:size(xdata,2)
            if final_1(i)>final_1(j) && final_2(i)>final_2(j)
                final_3(j) = final_3(j)+1;
            elseif final_1(i)<final_1(j) && final_2(i)<final_2(j)
                final_3(i) = final_3(i)+1;
            end
        end
    end
    [~,idx_final] = sort(final_3);
    idx_f = [idx_f;idx_final];
end

j = 1;
for i = 1:size(pop_all,1)
    if mse_test_all(ceil(i/10),i - ceil(i/10)*10 + 10) < mean(mean(mse_test_all))
        pop{j} = pop_all{i,:};
        mse(j) = mse_test_all(ceil(i/10),i - ceil(i/10)*10 + 10);
        j = j + 1;
    end
end
mse = mse';
mse = ones(100,1);
for i = 1:size(mse,1)
    ans = vpa(str2sym(cellstr(pop{i,:})),4);
    a = cellstr(string(char(ans)));
    b{i} = a;
end

for i = 1:size(mse,1)
    ans = vpa(str2sym(cellstr(pop{i,:})),2);
    a = cellstr(string(char(ans)));
    bb{i} = a;
end
% b = b';

r2 = 1-mean(mse)*mean(mse)/var(ydata);

for i = 1:1
    bd2 = bb{i};
    bd2 = strrep (bd2,'X15','xdata(:,15)');
    bd2 = strrep (bd2,'X14','xdata(:,14)');
    bd2 = strrep (bd2,'X13','xdata(:,13)');
    bd2 = strrep (bd2,'X12','xdata(:,12)');
    bd2 = strrep (bd2,'X11','xdata(:,11)');
    bd2 = strrep (bd2,'X10','xdata(:,10)');
    bd2 = strrep (bd2,'X9','xdata(:,9)');
    bd2 = strrep (bd2,'X8','xdata(:,8)');
    bd2 = strrep (bd2,'X7','xdata(:,7)');
    bd2 = strrep (bd2,'X6','xdata(:,6)');
    bd2 = strrep (bd2,'X5','xdata(:,5)');
    bd2 = strrep (bd2,'X4','xdata(:,4)');
    bd2 = strrep (bd2,'X3','xdata(:,3)');
    bd2 = strrep (bd2,'X2','xdata(:,2)');
    bd2 = strrep (bd2,'X1','xdata(:,1)');
    bd2 = strrep (bd2,'*','.*');
    bd2 = strrep (bd2,'/','./');
    bd2 = strrep (bd2,'^','.^');
    vv4 = eval(bd2{1,1});
    v4(i) = sqrt(mean(abs(vv4-ydata).^2));
    bd2 = b{i};
    bd2 = strrep (bd2,'X15','xdata(:,15)');
    bd2 = strrep (bd2,'X14','xdata(:,14)');
    bd2 = strrep (bd2,'X13','xdata(:,13)');
    bd2 = strrep (bd2,'X12','xdata(:,12)');
    bd2 = strrep (bd2,'X11','xdata(:,11)');
    bd2 = strrep (bd2,'X10','xdata(:,10)');
    bd2 = strrep (bd2,'X9','xdata(:,9)');
    bd2 = strrep (bd2,'X8','xdata(:,8)');
    bd2 = strrep (bd2,'X7','xdata(:,7)');
    bd2 = strrep (bd2,'X6','xdata(:,6)');
    bd2 = strrep (bd2,'X5','xdata(:,5)');
    bd2 = strrep (bd2,'X4','xdata(:,4)');
    bd2 = strrep (bd2,'X3','xdata(:,3)');
    bd2 = strrep (bd2,'X2','xdata(:,2)');
    bd2 = strrep (bd2,'X1','xdata(:,1)');
    bd2 = strrep (bd2,'*','.*');
    bd2 = strrep (bd2,'/','./');
    bd2 = strrep (bd2,'^','.^');
    vv2 = eval(bd2{1,1});
    v2(i) = sqrt(mean(abs(vv4-ydata).^2));
end
% aaa = [];
% for i = 1:10
%     aaa = [aaa,aa{i,:}];
% end
% ap = zeros(1,size(xdata,2));
% for i = 1:size(aaa,2)
%     ap(aaa(i)) = ap(aaa(i))+1;
% end
% [~,idx] = sort(ap);
% scatter(final_1,final_2);
% [idx,weights] = relieff(xdata,ydata,10);