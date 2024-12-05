function SelfAttentionLayer(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k)
save_folder_output=[save_folder,'/SELFATTENTION'];
if exist([save_folder_output,'/selfattention_model.mat'])==0
mkdir(save_folder_output);
numHiddenUnits=100;
%     rng(0)
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','gpu',...
    'InitialLearnRate',learning_rate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',2, ...   %2个epoch后学习率更新
    'LearnRateDropFactor',0.5, ...
    'Shuffle','every-epoch',...  % 时间序列长度
    'MiniBatchSize',miniBatchSize,...
    'ValidationData',{Validate_xNorm,Validate_yNorm}, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',true,...
    'Plots','training-progress');

lgraph = layerGraph();
tempLayers = [
    sequenceInputLayer([numFeatures 1 1],"Name","sequence")
    sequenceFoldingLayer("Name","seqfold")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer(kernel_size1,channel,"Name","conv1","Padding","same")
    batchNormalizationLayer("Name","batchnorm1")
    eluLayer(1,"Name","elu")
    maxPooling2dLayer(kernel_size2,"Name","maxpool","Padding","same")
    dropoutLayer(0.01,"Name","dropout1")
    batchNormalizationLayer("Name","batchnorm2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer(kernel_size1,channel,"DilationFactor",1,"Name","conv2","Padding","same")%,'Stride',[1,1]
    layerNormalizationLayer("Name","layernorm1")
    dropoutLayer(0.01,"Name","dropout2")
    convolution2dLayer(kernel_size2,channel,"DilationFactor",2,"Name","conv3","Padding","same")%'Stride',[2,2],
    layerNormalizationLayer("Name","layernorm2")
    eluLayer(1,"Name","elu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(2,2,"Name","concat1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    sequenceUnfoldingLayer("Name","sequnfold")
    flattenLayer("Name","flatten")
    bilstmLayer(numHiddenUnits,'Outputmode','sequence','name','bilstm_sequence')
    dropoutLayer(0.3,"Name","dropout")
    bilstmLayer(numHiddenUnits,'Outputmode','last','name','bilstm_last')    
    dropoutLayer(0.3,"Name","dropout_1")
    selfAttentionLayer(3,6)%,"BiasInitializer","narrow-normal","BiasL2Factor",0.0001,"WeightL2Factor",0.0001)
    fullyConnectedLayer(1,"Name","fc")
    tanhLayer("Name","tanh")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;
lgraph = connectLayers(lgraph,"seqfold/out","conv1");
lgraph = connectLayers(lgraph,"seqfold/out","conv2");
lgraph = connectLayers(lgraph,"seqfold/miniBatchSize","sequnfold/miniBatchSize");
lgraph = connectLayers(lgraph,"batchnorm2","concat1/in1");
lgraph = connectLayers(lgraph,"elu_1","concat1/in2");
lgraph = connectLayers(lgraph,"concat1","sequnfold/in");

plot(lgraph);
%
% 网络训练
% tic
selfattention_model = trainNetwork(Train_xNorm,Train_yNorm,lgraph,options);
save([save_folder_output,'/selfattention_model'],'selfattention_model');
else
    load ([save_folder_output,'/selfattention_model'],'selfattention_model')
end
%预测
YTrain_Pred =selfattention_model.predict(Train_xNorm);%YTrain_Pred=double(YTrain_Pred);
YTest_Pred = selfattention_model.predict(Test_xNorm);%YTest_Pred=double(YTest_Pred);
YValid_Pred = selfattention_model.predict(Validate_xNorm);%YValid_Pred=double(YValid_Pred);
%% 反归一化
%预测值反归一化
train_y=mapminmax('reverse',Train_yNorm',yopt);
test_y=mapminmax('reverse',Test_yNorm',yopt);
valide_y=mapminmax('reverse',Validate_yNorm',yopt);
YTrain_Pred=mapminmax('reverse',YTrain_Pred',yopt);
YTest_Pred=mapminmax('reverse',YTest_Pred',yopt);
YValid_Pred=mapminmax('reverse',YValid_Pred',yopt);
% save result/lstm YTrain_Pred YTest_Pred YValid_Pred
%%
% 画图
model_name='CNNBiLSTM-SelfAttention';
[train_evaluate,test_evaluate,validate_evaluate]=result_plot(save_folder_output,dataset_name,target_name,model_name,train_y,YTrain_Pred,test_y,YTest_Pred,valide_y,YValid_Pred);
evaluate=[train_evaluate;test_evaluate;validate_evaluate];
% 保存预测结果
if exist([save_folder_output,'/selfattention_model'])==0
save([save_folder_output,'/target_value'],"train_y","test_y","valide_y","YTrain_Pred","YTest_Pred","YValid_Pred");
% 保留误差结果
save([save_folder_output,'/evaluate_indicator'],"train_evaluate","test_evaluate","validate_evaluate","evaluate");
end
