function nse = NSE(sim, obs)        %输入参数分别为模拟值、实测值（序列）
ave_obs = mean(obs);   %实测数据平均数
Numerator = sum((obs-sim).^2);  %分子
Denominator = sum((obs-ave_obs).^2);%分母
nse = 1 - Numerator/Denominator;