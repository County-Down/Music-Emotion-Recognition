function DWSAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block)
save_folder_output=[save_folder,'/dwsam'];
train=exist([save_folder_output,'/dwsam_model.mat']);
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

for i=1:num_block
    dilationFactor = 2 ^(i - 1); 
    tempLayers = [
        convolution2dLayer(kernel_size1,channel,"DilationFactor",dilationFactor,"Name","convspace_"+i,"Padding","same")
        batchNormalizationLayer()
        eluLayer(1)
        maxPooling2dLayer(kernel_size2,"Name","maxpool1_"+i,"Padding","same")
        dropoutLayer(0.01)
        batchNormalizationLayer("name","batchnormspace_"+i)];
    lgraph = addLayers(lgraph,tempLayers);
    tempLayers = [
        convolution2dLayer(kernel_size1,channel,"DilationFactor",dilationFactor,"Name","convtime1_"+i,"Padding","same")
        layerNormalizationLayer()
        dropoutLayer(0.01)
        convolution2dLayer(kernel_size2,channel,"DilationFactor",dilationFactor,"Name","convtime2_"+i,"Padding","same")
        layerNormalizationLayer()
        eluLayer(1,"Name","elutime_"+i)];
    lgraph = addLayers(lgraph,tempLayers);
    tempLayers =[depthConcatenationLayer(2,"Name","concat"+i)
        ];
    lgraph = addLayers(lgraph,tempLayers);
    tempLayers =[additionLayer(2,"name","add_"+i)];
    lgraph = addLayers(lgraph,tempLayers);
    lgraph = connectLayers(lgraph,"batchnormspace_"+i,"concat"+i+"/in1");
    lgraph = connectLayers(lgraph,"elutime_"+i,"concat"+i+"/in2");
   % 注意力机制部分
   % w注意力机制
    tempLayers = globalAveragePooling2dLayer("Name","w_gapool_"+i);
    lgraph = addLayers(lgraph,tempLayers);
    tempLayers = globalMaxPooling2dLayer("Name","w_gmpool_"+i);
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = [
        depthConcatenationLayer(2,"Name","w_dconcat"+i)
        convolution2dLayer([1 1],channel*2,"Name","w_conv_dot_1"+i,"Padding","same")
        eluLayer(1)
        waveletLayer("waveletlayer"+i)
        convolution2dLayer([1 1],channel*2,"Name","w_conv_dot_2"+i,"Padding","same")
        sigmoidLayer("Name","w_sigmoid_"+i)];
    lgraph = addLayers(lgraph,tempLayers);
    % s注意力机制
    tempLayers = transposeLayer("s_transpose_1"+i);
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = globalAveragePooling2dLayer("Name","s_gapool_"+i);
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = globalMaxPooling2dLayer("Name","s_gmpool_"+i);
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = [
        concatenationLayer(1,2,"Name","s_concat"+i)
        transposeLayer("s_transpose_2"+i)
        convolution2dLayer([1 1],channel*2,"Name","s_conv_dot"+i,"Padding","same")
        sigmoidLayer("Name","s_sigmoid_"+i)];
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = [multiplicationLayer(3,"Name","multiplication"+i)];
    lgraph = addLayers(lgraph,tempLayers);
    
    % 添加w注意力机制
    lgraph = connectLayers(lgraph,"concat"+i,"w_gapool_"+i);
    lgraph = connectLayers(lgraph,"concat"+i,"w_gmpool_"+i);
    lgraph = connectLayers(lgraph,"w_gapool_"+i,"w_dconcat"+i+"/in1");
    lgraph = connectLayers(lgraph,"w_gmpool_"+i,"w_dconcat"+i+"/in2");
    % 添加s注意力机制
    lgraph = connectLayers(lgraph,"concat"+i,"s_transpose_1"+i);
    lgraph = connectLayers(lgraph,"s_transpose_1"+i,"s_gapool_"+i);
    lgraph = connectLayers(lgraph,"s_transpose_1"+i,"s_gmpool_"+i);
    lgraph = connectLayers(lgraph,"s_gmpool_"+i,"s_concat"+i+"/in1");
    lgraph = connectLayers(lgraph,"s_gapool_"+i,"s_concat"+i+"/in2");
    % 结果相乘
    lgraph = connectLayers(lgraph,"w_sigmoid_"+i,"multiplication"+i+"/in1");
    lgraph = connectLayers(lgraph,"s_sigmoid_"+i,"multiplication"+i+"/in2");
    lgraph = connectLayers(lgraph,"concat"+i,"multiplication"+i+"/in3");
    lgraph = connectLayers(lgraph,"multiplication"+i,"add_"+i+"/in1");
    % 添加残差
    if i == 1
        outputName="seqfold/out";
        lgraph = connectLayers(lgraph,"seqfold/out","convspace_1");
        lgraph = connectLayers(lgraph,"seqfold/out","convtime1_1");
        layer = convolution2dLayer(1, channel*2, Name = "convSkip");    % 建立残差卷积层
        lgraph = addLayers(lgraph, layer);                               % 将残差卷积层加入到网络
        lgraph = connectLayers(lgraph, outputName, "convSkip"); 
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
        
    else
        lgraph = connectLayers(lgraph,"add_" + (i-1),"convspace_"+i);
        lgraph = connectLayers(lgraph,"add_" + (i-1),"convtime1_"+i);
        lgraph = connectLayers(lgraph, "add_" + (i-1), "add_" + i+"/in2"); % 将残差层连接到 addtion 层 输入口2
    end

end



% 添加bilstm
tempLayers = [
    sequenceUnfoldingLayer("Name","sequnfold")
    flattenLayer("Name","flatten")
    bilstmLayer(numHiddenUnits,'Outputmode','sequence','name','bilstm_sequence')
    dropoutLayer(0.3,"Name","dropout1")
    bilstmLayer(numHiddenUnits,'Outputmode','last','name','bilstm_last')
    dropoutLayer(0.3,"Name","dropout2")
    fullyConnectedLayer(numResponses,"Name","fc")
    tanhLayer("Name","tanh")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

lgraph = connectLayers(lgraph,"seqfold/miniBatchSize","sequnfold/miniBatchSize");
lgraph = connectLayers(lgraph,"add_"+num_block,"sequnfold/in");
analyzeNetwork(lgraph);
plot(lgraph);
%
% 网络训练
% tic
dwsam_model = trainNetwork(Train_xNorm,Train_yNorm,lgraph,options);
save([save_folder_output,'/dwsam_model'],'dwsam_model');
else
    load ([save_folder_output,'/dwsam_model'],'dwsam_model')
end
%预测
YTrain_Pred =dwsam_model.predict(Train_xNorm);%YTrain_Pred=double(YTrain_Pred);
YTest_Pred = dwsam_model.predict(Test_xNorm);%YTest_Pred=double(YTest_Pred);
YValid_Pred = dwsam_model.predict(Validate_xNorm);%YValid_Pred=double(YValid_Pred);
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
model_name='DCNBiLSTM-WSAM';
[train_evaluate,test_evaluate,validate_evaluate]=result_plot(save_folder_output,dataset_name,target_name,model_name,train_y,YTrain_Pred,test_y,YTest_Pred,valide_y,YValid_Pred);
evaluate=[train_evaluate;test_evaluate;validate_evaluate];
% 保存预测结果
if train==0
save([save_folder_output,'/target_value'],"train_y","test_y","valide_y","YTrain_Pred","YTest_Pred","YValid_Pred");
% 保留误差结果
save([save_folder_output,'/evaluate_indicator'],"train_evaluate","test_evaluate","validate_evaluate","evaluate");
% 保留超参数
hyperparameters_data=[maxEpochs;learning_rate;miniBatchSize;validationFrequency;numHiddenUnits;kernel_size1;kernel_size2];
save([save_folder_output,'/hyperparameters_data'],"hyperparameters_data");
end
