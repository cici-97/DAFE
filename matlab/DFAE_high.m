close all
clear
clc

load 'D:\ͼ����\����-ר��-����\DFAE��������\HSIs\abu-urban-1'
load 'D:\ͼ����\����-ר��-����\DFAE��������\data_ori\abu-urban-1_ori'
load 'D:\ͼ����\����-ר��-����\DFAE��������\data_low\abu-urban-1_low1'
savepath = 'D:\ͼ����\����-ר��-����\DFAE��������\data_high\abu-urban-1_high1';

% ori_new = data;  % HYDICE���β���188������û��ori_new
[row,col,b] = size(ori_new);
data_high_tem = zeros(row,col,b);
data_high_tem_dir = zeros(row,col,b);

%-------------------- ��һ��ͼ�� ----------------------
data_max = max(max(max(ori_new)));
data_min = min(min(min(ori_new)));
ori_new = (ori_new - data_min)/(data_max - data_min);

tic
%--------------��������˹������ȡ��Ƶ��Ϣ ------------
H5 = fspecial('laplacian');
for i = 1:1:b
    I = ori_new(:,:,i);  %ԭʼͼ
    edge = imfilter(I, H5,'replicate');
    data_high_tem(:,:,i) = edge;
    disp(i)
end
data_high_max = max(max(max(data_high_tem)));
data_high_min = min(min(min(data_high_tem)));
data_high = (data_high_tem - data_high_min)/(data_high_max - data_high_min);
toc
save(savepath,'data_high');

%------------------ ֱ����� ---------------------
tic
for i = 1:1:b
    I = ori_new(:,:,i);  %ԭʼͼ
    p = data_low(:,:,i);  %��Ƶͼ
    H = I - p;
    data_high_tem_dir(:,:,i) = H;
    disp(i)
end
toc

data_high_max = max(max(max(data_high_tem_dir)));
data_high_min = min(min(min(data_high_tem_dir)));
data_high_dir = (data_high_tem_dir - data_high_min)/(data_high_max - data_high_min);

%--------------------------- ��ʾͼ�� ----------------------------
band = 100;
figure()
subplot(1,4,1)
imagesc(ori_new(:,:,band))   %���Ա任��ʾͼ��
axis off,axis image 

subplot(1,4,2)
imagesc(data_low(:,:,band))   %���Ա任��ʾͼ��
axis off,axis image

subplot(1,4,3)
imagesc(data_high_dir(:,:,band))   %���Ա任��ʾͼ��
color_hot=colormap(hot);%��ɫͼ����ȡ
mycolor=[color_hot(:,3),color_hot(:,2),color_hot(:,1)];%Ҳ������fliplr()����������������ɫͨ��
colormap (mycolor),brighten(0),axis off,axis image    %��ʾͼ���������

subplot(1,4,4)
imagesc(data_high(:,:,band))   %���Ա任��ʾͼ��
color_hot=colormap(hot);%��ɫͼ����ȡ
mycolor=[color_hot(:,3),color_hot(:,2),color_hot(:,1)];%Ҳ������fliplr()����������������ɫͨ��
colormap (mycolor),brighten(0),axis off,axis image    %��ʾͼ���������

