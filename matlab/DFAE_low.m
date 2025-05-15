close all
clear
clc

% load 'D:\图像所\毕设-专利-论文\DFAE代码整理\HSIs\HYDICE';
% ori_new = data;

load 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_ori\abu-urban-5_ori';
savepath_low = 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_low\abu-urban-5_low1';

%------------------ 截取原始图像中的前188个波段，保证其维度一致（截取后的图像都在data_ori里面） ---------------------
% savepath_ori= 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_ori\sandiego1_ori';
% load 'D:\图像所\毕设-专利-论文\DFAE代码整理\HSIs\sandiego1' ;
% [row,col,b] = size(data);
% tic
% ori_new = zeros(row,col,188);
% for i = 1:1:188
%     I = data(:,:,i);
%     ori_new(:,:,i) = I;
%     disp(i)
% end
% toc
% save(savepath_ori,'ori_new');

%--------------------- 提取低频信息 ---------------------
alpha_t = 0.001;  % 论文里的α，另一个参数在函数muGIF里面epsr和epst是一致的
N_iter = 8;
mode = 2; 

[row,col,b] = size(ori_new);
ori_max = max(max(max(ori_new)));
ori_min = min(min(min(ori_new)));
ori_new = (ori_new - ori_min)/(ori_max - ori_min);
tic
[T,R] = muGIF(ori_new,ori_new,alpha_t,0,N_iter,mode);
toc

%--------------------- 保存结果 ---------------------
data_low = T;
save(savepath_low,'data_low');


%--------------------- 显示图像 ---------------------
figure()
imagesc(ori_new(:,:,50))   %线性变换显示图像
axis off,axis image    %显示图像比例合适

figure()
imagesc(data_low(:,:,50))   %线性变换显示图像
axis off,axis image    %显示图像比例合适

% figure()
% imagesc(map)   %线性变换显示图像
% axis off,axis image    %显示图像比例合适

