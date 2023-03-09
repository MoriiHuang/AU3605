clear;
clc;
% size的值根据生成模板的大小变化
% 5*5 卷积核的size为5
% 7*7 卷积核的size为7
% 11*11 卷积核的size为11
% 15*15 卷积核的size为15
% 19*19 卷积核的size为19
% 此处以 11*11卷积核为例
size=19;
start_=-(size-1)/2;
end_=(size-1)/2;
[X,Y] = meshgrid(start_:end_,start_:end_);
sigma = (size-1)/4;
h = exp(-(X.*X+Y.*Y)./(2*sigma^2))/(2*pi*sigma^2);
% 归一化
sumh = sum(h(:));
h= h/sumh;
% surf(X,Y,h)
