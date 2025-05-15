close all
clear
clc

load 'D:\图像所\毕设-专利-论文\DFAE代码整理\HSIs\abu-urban-1'
load 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_ori\abu-urban-1_ori'
load 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_low\abu-urban-1_low1'
savepath = 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_high\abu-urban-1_high1';

% ori_new = data;  % HYDICE波段不够188，所以没有ori_new
[row,col,b] = size(ori_new);
data_high_tem = zeros(row,col,b);
data_high_tem_dir = zeros(row,col,b);

%-------------------- 归一化图像 ----------------------
data_max = max(max(max(ori_new)));
data_min = min(min(min(ori_new)));
ori_new = (ori_new - data_min)/(data_max - data_min);

tic
%--------------用拉普拉斯算子提取高频信息 ------------
H5 = fspecial('laplacian');
for i = 1:1:b
    I = ori_new(:,:,i);  %原始图
    edge = imfilter(I, H5,'replicate');
    data_high_tem(:,:,i) = edge;
    disp(i)
end
data_high_max = max(max(max(data_high_tem)));
data_high_min = min(min(min(data_high_tem)));
data_high = (data_high_tem - data_high_min)/(data_high_max - data_high_min);
toc
save(savepath,'data_high');

%------------------ 直接相减 ---------------------
tic
for i = 1:1:b
    I = ori_new(:,:,i);  %原始图
    p = data_low(:,:,i);  %低频图
    H = I - p;
    data_high_tem_dir(:,:,i) = H;
    disp(i)
end
toc

data_high_max = max(max(max(data_high_tem_dir)));
data_high_min = min(min(min(data_high_tem_dir)));
data_high_dir = (data_high_tem_dir - data_high_min)/(data_high_max - data_high_min);

%--------------------------- 显示图像 ----------------------------
band = 100;
figure()
subplot(1,4,1)
imagesc(ori_new(:,:,band))   %线性变换显示图像
axis off,axis image 

subplot(1,4,2)
imagesc(data_low(:,:,band))   %线性变换显示图像
axis off,axis image

subplot(1,4,3)
imagesc(data_high_dir(:,:,band))   %线性变换显示图像
color_hot=colormap(hot);%颜色图的提取
mycolor=[color_hot(:,3),color_hot(:,2),color_hot(:,1)];%也可以用fliplr()函数，交换红蓝颜色通道
colormap (mycolor),brighten(0),axis off,axis image    %显示图像比例合适

subplot(1,4,4)
imagesc(data_high(:,:,band))   %线性变换显示图像
color_hot=colormap(hot);%颜色图的提取
mycolor=[color_hot(:,3),color_hot(:,2),color_hot(:,1)];%也可以用fliplr()函数，交换红蓝颜色通道
colormap (mycolor),brighten(0),axis off,axis image    %显示图像比例合适

