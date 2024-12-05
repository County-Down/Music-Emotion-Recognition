function EVA=result(true_value,predict_value,type)
disp(type)
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['根均方差(RMSE)：',num2str(rmse)])
mae=mean(abs(true_value-predict_value));
disp(['平均绝对误差（MAE）：',num2str(mae)])
% 剔除为0的元素
[row,~]=find(true_value~=0);
mape=mean(abs((true_value(row)-predict_value(row))./true_value(row)));
disp(['平均相对百分误差（MAPE）：',num2str(mape*100),'%'])
r2 = R2(predict_value, true_value);
disp(['R平方决定系数：',num2str(r2)])
nse = NSE(predict_value, true_value);
disp(['纳什系数（NSE）：',num2str(nse)])
EVA=[rmse,mae,mape,r2];
fprintf('\n')