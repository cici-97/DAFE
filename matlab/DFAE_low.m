close all
clear
clc

% load 'D:\ͼ����\����-ר��-����\DFAE��������\HSIs\HYDICE';
% ori_new = data;

load 'D:\ͼ����\����-ר��-����\DFAE��������\data_ori\abu-urban-5_ori';
savepath_low = 'D:\ͼ����\����-ר��-����\DFAE��������\data_low\abu-urban-5_low1';

%------------------ ��ȡԭʼͼ���е�ǰ188�����Σ���֤��ά��һ�£���ȡ���ͼ����data_ori���棩 ---------------------
% savepath_ori= 'D:\ͼ����\����-ר��-����\DFAE��������\data_ori\sandiego1_ori';
% load 'D:\ͼ����\����-ר��-����\DFAE��������\HSIs\sandiego1' ;
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

%--------------------- ��ȡ��Ƶ��Ϣ ---------------------
alpha_t = 0.001;  % ������Ħ�����һ�������ں���muGIF����epsr��epst��һ�µ�
N_iter = 8;
mode = 2; 

[row,col,b] = size(ori_new);
ori_max = max(max(max(ori_new)));
ori_min = min(min(min(ori_new)));
ori_new = (ori_new - ori_min)/(ori_max - ori_min);
tic
[T,R] = muGIF(ori_new,ori_new,alpha_t,0,N_iter,mode);
toc

%--------------------- ������ ---------------------
data_low = T;
save(savepath_low,'data_low');


%--------------------- ��ʾͼ�� ---------------------
figure()
imagesc(ori_new(:,:,50))   %���Ա任��ʾͼ��
axis off,axis image    %��ʾͼ���������

figure()
imagesc(data_low(:,:,50))   %���Ա任��ʾͼ��
axis off,axis image    %��ʾͼ���������

% figure()
% imagesc(map)   %���Ա任��ʾͼ��
% axis off,axis image    %��ʾͼ���������

