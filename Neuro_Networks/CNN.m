function CNN(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k)
Train_xNorm=cnncell2flatten(Train_xNorm);
Test_xNorm=cnncell2flatten(Test_xNorm);
Validate_xNorm=cnncell2flatten(Validate_xNorm);
save_folder_output=[save_folder,'/cnn'];
train=exist([save_folder_output,'/cnn_model.mat']);
if train==0
    mkdir(save_folder_output);
%     rng(0)
    numFeatures = size(Train_xNorm,1);%输入节点数
    numResponses = size(Train_yNorm,2);%输出节点数
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
%'ValidationData',{Test_xNorm,Test_yNorm}, ...
    %'ValidationFrequency',validationFrequency, ...
lgraph = layerGraph();
tempLayers = [
    imageInputLayer([numFeatures 1 1],"Name","sequence")
    %sequenceFoldingLayer("Name","seqfold")
    ];
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

tempLayers = [
    concatenationLayer(2,2,"Name","concat1")
    %sequenceUnfoldingLayer("Name","sequnfold")
    %flattenLayer("Name","fl")
    fullyConnectedLayer(1,"Name","fc")
    tanhLayer("Name","tanh")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;
lgraph = connectLayers(lgraph,"sequence","conv1");
lgraph = connectLayers(lgraph,"sequence","conv2");
lgraph = connectLayers(lgraph,"batchnorm2","concat1/in1");
lgraph = connectLayers(lgraph,"elu_1","concat1/in2");
%lgraph = connectLayers(lgraph,'seqfold/miniBatchSize','sequnfold/miniBatchSize');
analyzeNetwork(lgraph);
plot(lgraph);
cnn_model = trainNetwork(Train_xNorm,Train_yNorm,lgraph,options);
save([save_folder_output,'/cnn_model'],'cnn_model');
else
    load ([save_folder_output,'/cnn_model'],'cnn_model')
end
%预测
YTrain_Pred =cnn_model.predict(Train_xNorm);%YTrain_Pred=double(YTrain_Pred);
YTest_Pred = cnn_model.predict(Test_xNorm);%YTest_Pred=double(YTest_Pred);
YValid_Pred = cnn_model.predict(Validate_xNorm);%YValid_Pred=double(YValid_Pred);
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
model_name='cnn_model';
[train_evaluate,test_evaluate,validate_evaluate]=result_plot(save_folder_output,dataset_name,target_name,model_name,train_y,YTrain_Pred,test_y,YTest_Pred,valide_y,YValid_Pred);
evaluate=[train_evaluate;test_evaluate;validate_evaluate];
% 保存预测结果
if train==0
save([save_folder_output,'/dataset','_',model_name],"Train_xNorm","Test_xNorm","Validate_xNorm","Train_yNorm","Test_yNorm","Validate_yNorm");    
save([save_folder_output,'/target_value','_',model_name],"train_y","test_y","valide_y","YTrain_Pred","YTest_Pred","YValid_Pred");
% 保留误差结果
save([save_folder_output,'/evaluate_indicator','_',model_name],"train_evaluate","test_evaluate","validate_evaluate","evaluate");
end
