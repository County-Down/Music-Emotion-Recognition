clc;clear;close all;format compact
% 全局变量
% 运行数据集与目标变量的名称
dataset_name='PMEmo';
target_name='Valence';
%导入数据集

target_data=table2array(readtable(['./',dataset_name,'/Dataset/',dataset_name,'/dynamic_annotations.csv']));

frameTime_count_all=xlsread(['./',dataset_name,'/frameTime_count/frameTime_count_all.xlsx']);

if strcmp(target_name,'Arousal')
target=target_data(:,end-1);

dynamic_features_analyze=xlsread(['./',dataset_name,'/randomforest/Arousal(mean)/dynamic_features_analyze.xlsx']);
dynamic_features_analyze=dynamic_features_analyze(:,2:end);
clear target_data;
%网络参数
kernel_size1=10;
kernel_size2=5;

else 
target=target_data(:,end);
dynamic_features_analyze=xlsread(['./',dataset_name,'/randomforest/Valence(mean)/dynamic_features_analyze.xlsx']);
dynamic_features_analyze=dynamic_features_analyze(:,2:end);
clear target_data;
%网络参数
kernel_size1=15;
kernel_size2=6;
end
% 保存路径
save_folder=['./',dataset_name,'_result','/',target_name];
mkdir(save_folder);

% 数据集划分
miniBatchSize = 168; %batchsize
channel=20;%10;
k=26;
block_num=3;
%dynamic_features_analyze=table2array(featuresanalyzearousal);
numFeatures=size(dynamic_features_analyze,2);

[Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt]=dataset_crossvalind(save_folder,target,frameTime_count_all,dynamic_features_analyze,k,numFeatures);
% 网络参数
numFeatures = size(Train_xNorm{1},1);%输入节点数
numResponses = size(Train_yNorm,2);%输出节点数
numHiddenUnits =50;
maxEpochs=40;%40
learning_rate=0.004;
validationFrequency = floor(size(Train_yNorm,1)/miniBatchSize/2);
% 神经网络训练
% 研究的网络
DSelfAttention(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,block_num);


% % 消融实验
% % cnn部分
CNN(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% bilstm部分
BiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm
CNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm时间尺度
TCNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm空间尺度
SCNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn部分添加残差
DCNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,block_num);

% cnn添加wsam注意力机制
DWSAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block);
% cnn添加se+空间注意力机制
DSESAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block);
% cnn添加cbam注意力机制
DCBAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block);