function SE_WSAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k)
save_folder_output=[save_folder,'/se_wsam'];
train=exist([save_folder_output,'/se_wsam_model.mat']);
% channel=channel+5;      
if train==0
mkdir(save_folder_output);
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
    sequenceInputLayer([numFeatures,1,1],"Name","sequence")
    sequenceFoldingLayer("Name","seqfold")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer(kernel_size1,channel,'name','conv1',"Padding","same")%'Stride',[1,1],
    batchNormalizationLayer("Name","batchnorm1")
    eluLayer(1,"Name","elu")
    maxPooling2dLayer(kernel_size2,'Padding','same','name','maxpool')%'Stride',2,
    dropoutLayer(0.01,"Name","dropout1")
    batchNormalizationLayer("Name","batchnorm2")
    ];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer(kernel_size1,channel,"DilationFactor",1,"Name","conv2","Padding","same")%,'Stride',[1,1]
    layerNormalizationLayer("Name","layernorm1")
    dropoutLayer(0.01,"Name","dropout2")
    convolution2dLayer(kernel_size2,channel,"DilationFactor",2,"Name","conv3","Padding","same")%'Stride',[2,2],
    layerNormalizationLayer("Name","layernorm2")
    eluLayer(1,"Name","elu_1")
    ];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(2,2,"Name","concat1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","gapool");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalMaxPooling2dLayer("Name","gmpool");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(2,2,"Name","concat2")
    eluLayer(1,"Name","elu_2")
    fullyConnectedLayer(channel,"Name","fc_s")
    waveletLayer('wavelet')
    convolution2dLayer([1 1],channel,"Name","conv_dot_e","Padding","same")
    %layerNormalizationLayer("Name","layernorm_3")
    %fullyConnectedLayer(10,"Name","fc_e")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = transposeLayer("transpose_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","gapool_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalMaxPooling2dLayer("Name","gmpool_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(1,2,"Name","concat_1")
    transposeLayer("transpose_2")
    convolution2dLayer([1 1],1,'Stride',[1,1],"Name","conv_dot","Padding","same")
    sigmoidLayer("Name","sigmoid_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(3,"Name","multiplication")
    sequenceUnfoldingLayer("Name","sequnfold")
    flattenLayer("Name","flatten")
    % tswitchLayer('t2s')
    % stftLayer()
    % squeezeLayer("sq")
    % flattenLayer("Name","flatten1")
    %fullyConnectedLayer(numHiddenUnits1*2,"Name","fc1")
    bilstmLayer(numHiddenUnits,'Outputmode','sequence','name','bilstm_sequence')
    dropoutLayer(0.3,"Name","dropout")
    %RBFLayer('wnn')
    %eluLayer(1,"Name","elu_3")
    bilstmLayer(numHiddenUnits,'Outputmode','last','name','bilstm_last')
    dropoutLayer(0.3,"Name","dropout_1")
    %selfAttentionLayer(1,1,"Name","selfattention")
    %RBFLayer('wnn')
    fullyConnectedLayer(numResponses,"Name","fc")
    tanhLayer("Name","tanh")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;
lgraph = connectLayers(lgraph,"seqfold/out","conv1");
lgraph = connectLayers(lgraph,"seqfold/out","conv2");
lgraph = connectLayers(lgraph,"batchnorm2","concat1/in1");
lgraph = connectLayers(lgraph,"elu_1","concat1/in2");
lgraph = connectLayers(lgraph,"concat1","gapool");
lgraph = connectLayers(lgraph,"concat1","gmpool");
lgraph = connectLayers(lgraph,"concat1","transpose_1");
lgraph = connectLayers(lgraph,"concat1","multiplication/in1");
lgraph = connectLayers(lgraph,"gapool","concat2/in1");
lgraph = connectLayers(lgraph,"gmpool","concat2/in2");
lgraph = connectLayers(lgraph,"sigmoid","multiplication/in2");
lgraph = connectLayers(lgraph,"transpose_1","gapool_1");
lgraph = connectLayers(lgraph,"transpose_1","gmpool_1");
lgraph = connectLayers(lgraph,"gapool_1","concat_1/in1");
lgraph = connectLayers(lgraph,"gmpool_1","concat_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1","multiplication/in3");
lgraph = connectLayers(lgraph,'seqfold/miniBatchSize','sequnfold/miniBatchSize');
plot(lgraph);
analyzeNetwork(lgraph);
% 网络训练
% tic
se_wsam_model = trainNetwork(Train_xNorm,Train_yNorm,lgraph,options);
save([save_folder_output,'/se_wsam_model'],'se_wsam_model');
else
    load ([save_folder_output,'/se_wsam_model'],'se_wsam_model');
end
%预测
YTrain_Pred =se_wsam_model.predict(Train_xNorm);%YTrain_Pred=double(YTrain_Pred);
YTest_Pred = se_wsam_model.predict(Test_xNorm);%YTest_Pred=double(YTest_Pred);
YValid_Pred = se_wsam_model.predict(Validate_xNorm);%YValid_Pred=double(YValid_Pred);
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
model_name='CNNBiLSTM-SE-WSAM';
[train_evaluate,test_evaluate,validate_evaluate]=result_plot(save_folder_output,dataset_name,target_name,model_name,train_y,YTrain_Pred,test_y,YTest_Pred,valide_y,YValid_Pred);
evaluate=[train_evaluate;test_evaluate;validate_evaluate];
% 保存预测结果
if train==0
save([save_folder_output,'/target_value'],"train_y","test_y","valide_y","YTrain_Pred","YTest_Pred","YValid_Pred");
% 保留误差结果
save([save_folder_output,'/evaluate_indicator'],"train_evaluate","test_evaluate","validate_evaluate","evaluate");
end
