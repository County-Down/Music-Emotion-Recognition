clc;clear;close all;format compact

% Run the dataset with the name of the target variable
dataset_name='PMEmo';
target_name='Valence';
% Importing the dataset

target_data=table2array(readtable(['./',dataset_name,'/Dataset/',dataset_name,'/dynamic_annotations.csv']));

frameTime_count_all=xlsread(['./',dataset_name,'/frameTime_count/frameTime_count_all.xlsx']);

if strcmp(target_name,'Arousal')
target=target_data(:,end-1);

dynamic_features_analyze=xlsread(['./',dataset_name,'/randomforest/Arousal(mean)/dynamic_features_analyze.xlsx']);
dynamic_features_analyze=dynamic_features_analyze(:,2:end);
clear target_data;

%Network parameters
kernel_size1=10;
kernel_size2=5;

else 
target=target_data(:,end);
dynamic_features_analyze=xlsread(['./',dataset_name,'/randomforest/Valence(mean)/dynamic_features_analyze.xlsx']);
dynamic_features_analyze=dynamic_features_analyze(:,2:end);
clear target_data;
%Network parameters
kernel_size1=15;
kernel_size2=6;
end
%save path
save_folder=['./',dataset_name,'_result','/',target_name];
mkdir(save_folder);

% traing parameters
miniBatchSize = 168; %batchsize
channel=20;%10;
k=26;
block_num=3;
%dynamic_features_analyze=table2array(featuresanalyzearousal);
numFeatures=size(dynamic_features_analyze,2);

%Build a time series model
[Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt]=dataset_crossvalind(save_folder,target,frameTime_count_all,dynamic_features_analyze,k,numFeatures);

% traing parameters
numFeatures = size(Train_xNorm{1},1);%输入节点数
numResponses = size(Train_yNorm,2);%输出节点数
numHiddenUnits =50;
maxEpochs=40;%40
learning_rate=0.004;
validationFrequency = floor(size(Train_yNorm,1)/miniBatchSize/2);

% Neural Network Training
% OURS
DSelfAttention(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,block_num);


% % ablation study
% % cnn
CNN(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% bilstm
BiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm(No Residual block)
CNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm-Scale 1(No Residual block)
TCNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm-Scale 2(No Residual block)
SCNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm-Residual-block
DCNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,block_num);

% Attention Mechanism
% SE+SPATIAL Attention Mechanism
DSESAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block);
% CBAM Attention Mechanism
DCBAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block);
