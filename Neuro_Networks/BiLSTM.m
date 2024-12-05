function BiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k)
Train_xNorm=bilstm_input(Train_xNorm,k);
Test_xNorm=bilstm_input(Test_xNorm,k);
Validate_xNorm=bilstm_input(Validate_xNorm,k);
save_folder_output=[save_folder,'/bilstm'];
train=exist([save_folder_output,'/bilstm_model.matl']);
if train==0
mkdir(save_folder_output);

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','gpu',...
    'InitialLearnRate',learning_rate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',2, ...   %2个epoch后学习率更新
    'LearnRateDropFactor',0.5, ...
    'Shuffle','every-epoch',...  % 时间序列长度
    'ValidationData',{Validate_xNorm,Validate_yNorm}, ...
    'ValidationFrequency',validationFrequency, ...
    'MiniBatchSize',miniBatchSize,...
    'Verbose',true,...
    'Plots','training-progress');
numFeatures=size(Train_xNorm{1},1);
lgraph = layerGraph();
tempLayers = [
    sequenceInputLayer(numFeatures,"Name","sequence")
    %flattenLayer("Name","flatten")
    bilstmLayer(numHiddenUnits,'Outputmode','sequence','name','bilstm_sequence')
    dropoutLayer(0.3,"Name","dropout")
    bilstmLayer(numHiddenUnits,'Outputmode','last','name','bilstm_last')
    dropoutLayer(0.3,"Name","dropout_1")
    fullyConnectedLayer(numResponses,"Name","fc")
    tanhLayer("Name","tanh")
    flattenLayer("Name","flatten1")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);
%analyzeNetwork(lgraph);
% 清理辅助变量
clear tempLayers;
plot(lgraph);
bilstm_model = trainNetwork(Train_xNorm,Train_yNorm,lgraph,options);
save([save_folder_output,'/bilstm_model'],'bilstm_model');
else
    load ([save_folder_output,'/bilstm_model'],'bilstm_model')
end
%预测
YTrain_Pred =bilstm_model.predict(Train_xNorm);%YTrain_Pred=double(YTrain_Pred);
YTest_Pred = bilstm_model.predict(Test_xNorm);%YTest_Pred=double(YTest_Pred);
YValid_Pred = bilstm_model.predict(Validate_xNorm);%YValid_Pred=double(YValid_Pred);
%% 反归一化
%预测值反归一化
train_y=mapminmax('reverse',Train_yNorm,yopt);
test_y=mapminmax('reverse',Test_yNorm,yopt);
valide_y=mapminmax('reverse',Validate_yNorm,yopt);
YTrain_Pred=mapminmax('reverse',YTrain_Pred,yopt);
YTest_Pred=mapminmax('reverse',YTest_Pred,yopt);
YValid_Pred=mapminmax('reverse',YValid_Pred,yopt);
% save result/lstm YTrain_Pred YTest_Pred YValid_Pred
%%
% 画图
model_name='bilstm_model';
[train_evaluate,test_evaluate,validate_evaluate]=result_plot(save_folder_output,dataset_name,target_name,model_name,train_y,YTrain_Pred,test_y,YTest_Pred,valide_y,YValid_Pred);
evaluate=[train_evaluate;test_evaluate;validate_evaluate];
% 保存预测结果
if train==0
save([save_folder_output,'/dataset','_',model_name],"Train_xNorm","Test_xNorm","Validate_xNorm","Train_yNorm","Test_yNorm","Validate_yNorm");    
save([save_folder_output,'/target_value','_',model_name],"train_y","test_y","valide_y","YTrain_Pred","YTest_Pred","YValid_Pred");
% 保留误差结果
save([save_folder_output,'/evaluate_indicator','_',model_name],"train_evaluate","test_evaluate","validate_evaluate","evaluate");
end
