function [x,xconf,yconf]=conf(true_value)
rate=0.9;
% 数据集
x=1:length(true_value);
y=true_value;
xconf = [x x(end:-1:1)] ;%一个来回
% 置信度90%
yconf=y*(1-rate);
yconf = [y+yconf y(end:-1:1)-yconf];%0.15就是条带宽度，换成矩阵就会有不同的宽度
end