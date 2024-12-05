function  [train_evaluate,test_evaluate,validate_evaluate]=result_plot(save_folder,dataset_name,target_name,model_name,train_y,YTrain_Pred,test_y,YTest_Pred,valide_y,YValid_Pred)
disp('训练集结果分析')
train_evaluate=result(train_y',YTrain_Pred','train');
figure
plot(train_y')
hold on
plot(YTrain_Pred')
grid on
title(model_name)
legend('True-Value','Predict-Value')
xlabel('Train-Dataset')
ylabel([dataset_name,'-',target_name])
saveas(gcf, [save_folder,'/',dataset_name,'_',target_name,'_Train.png']);

disp('测试集结果分析')
test_evaluate=result(test_y',YTest_Pred','test');
figure
plot(test_y')
hold on
plot(YTest_Pred')
grid on
title(model_name)
legend('True-Value','Predict-Value')
xlabel('Test-Dataset')
ylabel([dataset_name,'-',target_name])
saveas(gcf, [save_folder,'/',dataset_name,'_',target_name,'_Test.png']);

disp('验证集结果分析')
validate_evaluate=result(valide_y',YValid_Pred','valide');
figure
plot(valide_y')
hold on
plot(YValid_Pred')
grid on
title(model_name)
legend('True-value','Predict-value')
xlabel('Validation-Dataset')
ylabel([dataset_name,'-',target_name])
saveas(gcf, [save_folder,'/',dataset_name,'-',target_name,'_Validation.png']);

if model_name=="GLEAM"
% 置信度90%
rate=0.90;
% 训练集
[x_train,xconf_train,yconf_train]=conf(train_y);
figure
subplot(3,1,1)
fill(xconf_train,yconf_train,'r','FaceColor',[0.8 0.8 0.8],'EdgeColor','k');%FaceColor为填充颜色，EdgeColor为边框颜色
hold on;grid on
plot(x_train,train_y,'-.')
plot(x_train,YTrain_Pred,'--')
legend('90% confidence interval','True-value','Predict-value')
xlabel('Train-Dataset')
ylabel([dataset_name,'-',target_name])

% 测试集
[x_test,xconf_test,yconf_test]=conf(test_y);
subplot(3,1,2)
fill(xconf_test,yconf_test,'r','FaceColor',[0.8 0.8 0.8],'EdgeColor','k');%FaceColor为填充颜色，EdgeColor为边框颜色
hold on;grid on
plot(x_test,test_y,'-.')
plot(x_test,YTest_Pred,'--')
legend('90% confidence interval','True-value','Predict-value')
xlabel('Test-Dataset')
ylabel([dataset_name,'-',target_name])

[x_validate,xconf_validate,yconf_validate]=conf(valide_y);
subplot(3,1,3)
fill(xconf_validate,yconf_validate,'r','FaceColor',[0.8 0.8 0.8],'EdgeColor','k');%FaceColor为填充颜色，EdgeColor为边框颜色
hold on;grid on
plot(x_validate,valide_y,'-.')
plot(x_validate,YValid_Pred,'--')
legend('90% confidence interval','True-value','Predict-value')
xlabel('Validation-Dataset')
ylabel([dataset_name,'-',target_name])
sgtitle(model_name)
saveas(gcf, [save_folder,'/',dataset_name,'-',target_name,'_confidence_interval.png']);
end