clear;
clc;

size=41;
start_=-(size-1)/2;
end_=(size-1)/2;
[X,Y] = meshgrid(start_:end_,start_:end_);
%生成\sigma=5时的高斯曲面
sigma_5=5;
name="Gauss_Surface.png";
h = exp(-(X.*X+Y.*Y)./(2*sigma_5^2))/(2*pi*sigma_5^2);
% 归一化
sumh = sum(h(:));
h= h/sumh;
surf(X,Y,h);
drawnow
%生成\sigma从1变化到5时高斯曲面变化的gif
filename="Gauss_Surface.gif";
fig=figure;
for sigma =1:0.2:5
    h = exp(-(X.*X+Y.*Y)./(2*sigma^2))/(2*pi*sigma^2);
    % 归一化
    sumh = sum(h(:));
    h= h/sumh;
    surf(X,Y,h);
    xlabel('x');
    ylabel('y');
    zlabel('z');
    axis([-20 20 -20 20 0 0.25]);
    drawnow
    frame=getframe(fig);
    im=frame2im(frame);
    [imind,cm]=rgb2ind(im,256);
    if sigma == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
    end 
end