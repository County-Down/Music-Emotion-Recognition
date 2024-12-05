function [Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt]=dataset_crossvalind(save_folder,target,frameTime_count_all,dynamic_features_analyze,k,numFeatures)
if exist([save_folder,'/gleam_model.mat'])==0;
dynamic_features_frameTime_count=frameTime_count_all(:,end-1)';
Dynamic_frameTime_count=frameTime_count_all(:,end); 
%target=Arousal_mean(:,end);
%提取变量训练开始时的位置
start_step=dynamic_features_frameTime_count-Dynamic_frameTime_count'-k+1;
%dynamic_features_analyze=dynamic_features_analyze(:,2:end);
%求向量累加
dynamic_features_frameTime_count_cumsum=cumsum(dynamic_features_frameTime_count,2);
Dynamic_frameTime_count_cumsum=cumsum(Dynamic_frameTime_count,1);
% %训练集归一化
[dynamic_features_analyze_norm,xopt] = mapminmax(dynamic_features_analyze',0,1);
[target_norm,yopt] = mapminmax(target',0,1);
save yopt
% %按歌曲划分块
matrix_Data=mat2cell(dynamic_features_analyze_norm,[size(dynamic_features_analyze,2)],[dynamic_features_frameTime_count]);
matrix_Target=mat2cell(target_norm,1,[Dynamic_frameTime_count]);
Train_xNorm=[];
Test_xNorm=[];
Validate_xNorm=[];
Train_yNorm=[];
Test_yNorm=[];
Validate_yNorm=[];
%遍历每一首歌
for j=1:size(dynamic_features_frameTime_count,2)
    %初始化变量
    matirx_div=[];
    t1=matrix_Data{j};
    t2=matrix_Target{j};
    temp2=[];
    Train_xNorm1=[];
    Train_yNorm1=[];
    Test_xNorm1=[];
    Test_yNorm1=[];
    Validation_xNorm1=[];
    Validation_yNorm1=[];
    for i = start_step(j):dynamic_features_frameTime_count(j)-k
        temp2{1,i-start_step(j)+1} = reshape(t1(:,i:i+k-1),numFeatures,[],1,k);
    end
    matirx_div=[matirx_div,temp2];
    rng('default') % For reproducibility
    n1= size(temp2,2);
    [train1, test1] = crossvalind('holdOut',n1,0.2);
%     hpartition = cvpartition(n1,'Holdout',0.2); % Nonstratified partition
    %% 训练集
    %训练集自变量
%     Train_xNorm_id = training(hpartition);%生成用于划分训练集的id
    % 划分训练验证集
    Train_Validate_x= matirx_div(:,train1);
    Train_Validate_y=t2(:,train1);

    n2=size(Train_Validate_x,2);
    [train2, test2] = crossvalind('holdOut',n2,0.25);
    % 划分训练集
    Train_xNorm1=Train_Validate_x(:,train2);%用生成的id来划分训练集
    Train_xNorm=[Train_xNorm,Train_xNorm1];
    %训练集目标变量
    Train_yNorm1= Train_Validate_y(:,train2);%用生成的id来划分训练集
    Train_yNorm=[Train_yNorm,Train_yNorm1];
    %% 测试集
    %测试集自变量
%     Test_xNorm_id = test(hpartition);%生成用于划分训练集的id
    Test_xNorm1= matirx_div(:,test1);%用生成的id来划分训练集
    Test_xNorm=[Test_xNorm,Test_xNorm1];
    %测试因变量
    Test_yNorm1= t2(:,test1);%用生成的id来划分训练集
    Test_yNorm=[Test_yNorm,Test_yNorm1];
    %% 验证集
    %验证集自变量
    %rng('default')    
%     hpartition1 = cvpartition(n2,'Holdout',0.1); % Nonstratified partition
%     Validation_xNorm_id = test(hpartition1);%生成用于划分验证集的id
    Validation_xNorm1= Train_Validate_x(:,test2);%用生成的id来划分验证集
    Validate_xNorm=[Validate_xNorm,Validation_xNorm1];
    %验证集因变量
    Validation_yNorm1= Train_Validate_y(:,test2);%用生成的id来划分验证集
    Validate_yNorm=[Validate_yNorm,Validation_yNorm1];
end
Train_yNorm=Train_yNorm';
Test_yNorm=Test_yNorm';
Validate_yNorm=Validate_yNorm';
% save Data_set Train_xNorm Test_xNorm Validate_xNorm Train_yNorm Test_yNorm Validate_yNorm
% clear Arousal_mean frameTime_count_all dynamic_features_analyze
% 保存数据集划分结果
save([save_folder,'/data_set'],"Train_xNorm","Train_yNorm","Test_xNorm","Test_yNorm","Validate_xNorm","Validate_yNorm");
% 保存归一化矩阵
save([save_folder,'/minmax_opt'],"xopt","yopt")
else 
    load([save_folder,'/data_set']);
    load([save_folder,'/minmax_opt']);
end
