clear;
clc;
path =char(strcat(cd,filesep));
images = dir([path,'*.jpg']);
%resluts 用于记录  GaussBlur时卷积的结果
results=cell(3,5);
% GaussBlur

folder='./gaussblur_result/';

if ~exist(folder,'dir')
    mkdir(folder);
end
kernel_size=[5,7,11,15,19];
for image_index =1:length(images)
    for kernel_index=1:5
        size_=kernel_size(kernel_index);
        sigma=(size_-1)/4;
        start_=-(size_-1)/2;
        end_=(size_-1)/2;
        [X,Y] = meshgrid(start_:end_,start_:end_);
        h = exp(-(X.*X+Y.*Y)./(2*sigma^2))/(2*pi*sigma^2);
        % 归一化
        sumh = sum(h(:));
        h= h/sumh;
        % 用生成的高斯模板和图像卷积
        img = imread([path,images(image_index).name]);
        %记录卷积结果
        gauss_conved=conv2(h,img);
        results{image_index,kernel_index}=gauss_conved;
        %利用自带函数填充
        new_img = imfilter(img,h,"replicate");
        forename=strsplit(images(image_index).name,'.');
        endname = sprintf("_%dKernel.jpg",size_);
        filename=strcat('gaussblur_result',filesep,forename{1},endname);
        imwrite(new_img,filename);
    end
end

% Laplacian 
%其中resluts 为 GaussBlur时卷积的结果
folder='./laplacian_result/';
if ~exist(folder,'dir')
    mkdir(folder);
end
folder='./laplacian_strengthened/';
if ~exist(folder,'dir')
    mkdir(folder);
end
laplacian_kernel=[0 -1 0; -1 4 -1;0 -1 0];
for image_index =1:length(images)
    for kernel_index=1:5
        lap_conved=conv2(laplacian_kernel,results{image_index,kernel_index});
        forename=strsplit(images(image_index).name,'.');
        filename = strcat("laplacian_result",filesep,forename(1),'_', ...
            num2str(kernel_size(kernel_index)),'Kernel',"_laplacian.jpg");
        % 利用 im2uint8 归一化生成图像
        imwrite(im2uint8(lap_conved),filename);
        %利用自带函数填充
        lap_img=imfilter(results{image_index,kernel_index},laplacian_kernel,'replicate');
        stren_img=uint8(lap_img+results{image_index,kernel_index});
        filename = strcat("laplacian_strengthened",filesep,forename(1),'_', ...
            num2str(kernel_size(kernel_index)),'Kernel',"_laplacian.jpg");
        % 生成增强后的图像
        imwrite(stren_img,filename);
    end
end

% Gradient
%其中resluts 为 GaussBlur时卷积的结果
folder='./gradient_result/';
Prewitt_Kernel=[1,0,-1;1,0,-1;1,0,-1];
if ~exist(folder,'dir')
    mkdir(folder);
end
for image_index =1:length(images)
    for kernel_index=1:5
        gra_conved=conv2(Prewitt_Kernel,results{image_index,kernel_index});
        forename=strsplit(images(image_index).name,'.');
        filename = strcat("gradient_result",filesep,forename(1),'_', ...
            num2str(kernel_size(kernel_index)),'Kernel',"_gradient.jpg");
        % 利用uint8 生成图像
        imwrite(uint8(gra_conved),filename);
    end
end
