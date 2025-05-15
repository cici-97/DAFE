close all
clear
clc

area =55;   % 点异常设为15，结构异常设为55
load 'D:\图像所\毕设-专利-论文\DFAE代码整理\HSIs\sandiego1'  % 得到map，用于后面计算AUC时候使用
load 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_low_res\sandiego1_low'
load 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_high_res\sandiego1_high'
savepath_res = 'D:\图像所\毕设-专利-论文\DFAE代码整理\data_res\sandiego1';

tic
%---------------------------低频域属性滤波检测-----------------------------
f0_ori = double(low_res);
f0 = NormalizeData(f0_ori);
high = mat2gray(high_res);

% figure()
% subplot(2,3,1)
% imshow(f0(:,:,1))
% subplot(2,3,2)
% imshow(f0(:,:,2))
% subplot(2,3,3)
% imshow(f0(:,:,3))
% subplot(2,3,5)
% imshow(high)

d0 = morph_detect(f0,area,'a');
[row,col,bands]=size(d0);
output = zeros(row,col);
for i = 1:1:3
    img = d0(:,:,i);
    output = output + img;
end
output1 = output/3;
% figure()
% imshow(mat2gray(output1))   %线性变换显示图像
% axis off,axis image

%-----------------------------引导滤波去噪---------------------------------
output2 = mat2gray(output1);
r_h = 1;   
eps_h = 0.12^2;   %0.06 urban1训练   0.03
output3 = guidedfilter(output2, output2, r_h, eps_h);

r_h = 1;   
eps_h = 0.2^2;    % 2
output_h = guidedfilter(high, high, r_h, eps_h);
%--------------------------指数融合-----------------------------
D1 = mat2gray(output_h);
D0 = mat2gray(output3);
% D = D1.*D0;

a = 1;
exp_ori = -a*D1;
k = 1 - exp(exp_ori);
D = k.*D0;
toc

% 参数设置
% A_tem = 0; 
% % figure(),
% % hold on
% for a=0:0.1:1
%     exp_ori = -a*D1;
%     k = 1 - exp(exp_ori);
%     D = k.*D0;
%     D2 = mat2gray(D);
%     [PD0 PF0] = roc(map(:), D2(:));
%     area = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
% %     plot(a,area,'b*-');
% %     axis([1,20,0.95,1])  %确定x轴与y轴框图大小
%     if A_tem < area
%         A_tem = area
%         A = a
%     end
% end

p3 = ImGray2Pseudocolor(mat2gray(D), 'hot', 255);
high1 = ImGray2Pseudocolor(D1, 'hot', 255);
% high2 = ImGray2Pseudocolor(k, 'hot', 255);
output5 = ImGray2Pseudocolor(D0, 'hot', 255);
figure()
subplot(1,4,1)
imshow(high1)
xlabel('高频域')
subplot(1,4,2)
imshow(output5)
xlabel('低频域')
subplot(1,4,3)
imshow(p3)
xlabel('融合结果')
subplot(1,4,4)
imshow(map)
xlabel('map')

% figure()
% imshow(high1)
% figure()
% imshow(output5)
figure()
imshow(p3)
% ----------------------------- draw ROC精度 ----------------------------------------
figure('color',[1,1,1]),
hold on
[PD0 PF0] = roc(map(:), D0(:));
area_low = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
plot(PF0,PD0,'--','LineWidth',1.5)
text(0.0001,0.9,sprintf('AUC_ =%f',area_low));
[PD0 PF0] = roc(map(:), D1(:));
area_high = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
plot(PF0,PD0,'--','LineWidth',1.5)
text(0.0001,0.7,sprintf('AUC_ =%f',area_high));

figure('color',[1,1,1]),
hold on
D2 = mat2gray(D);
[PD0 PF0] = roc(map(:), D2(:));
plot(PF0,PD0,'--','LineWidth',1.5)
area = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
text(0.0001,0.9,sprintf('AUC_ =%f',area));
box on
xlabel('FPR','Fontname','Times New Roman','FontSize',10,'FontWeight','bold')
ylabel('TPR','Fontname','Times New Roman','FontSize',10,'FontWeight','bold')
set(gca,'xscale','log')
set(gca,'Fontname','Times New Roman','FontWeight','bold','Linewidth',1.5)
axis([1e-4 1 0 1])
% set(gcf,'position',[200,200,260,215])
set(gcf,'position',[200,200,300,250])

%---------------------保存结果---------------------------------
DFAE = D2;
save(savepath_res,'DFAE');

%---------------------draw ROC虚警---------------------------------
[row,col,bands]=size(data);
N=row * col; %N是高光谱图像的行乘列，即像素个数
targets = reshape(map, 1, N); %map是ground truth

outputs3 = reshape(D0, 1, N);%res是检测结果
[FPR0,TPR0] = myPlotROC(targets, outputs3);
x = linspace(1e-4,1,1003);
auc_low = trapz(x,FPR0);     %计算梯度面积积分
figure,plot(x,FPR0);
xlabel('false alarm rate');
ylabel('probability of detection');
title('ROC curves of detection algorithm');
text(0.5,0.7,sprintf('AUCROC_ =%f',auc_low));

outputs4 = reshape(D1, 1, N);
[FPR1,TPR1] = myPlotROC(targets, outputs4);
x = linspace(1e-4,1,1003);
auc_high = trapz(x,FPR1);     %计算梯度面积积分
figure,plot(x,FPR1);
xlabel('false alarm rate');
ylabel('probability of detection');
title('ROC curves of detection algorithm');
text(0.5,0.7,sprintf('AUCROC_ =%f',auc_high));

outputs1 = reshape(mat2gray(D), 1, N);   %融合结果
[FPR,TPR] = myPlotROC(targets, outputs1);
x = linspace(1e-4,1,1003);
auc3 = trapz(x,FPR);     %计算梯度面积积分
figure,plot(x,FPR);
xlabel('false alarm rate');
ylabel('probability of detection');
title('ROC curves of detection algorithm');
text(0.5,0.7,sprintf('AUCROC_ =%f',auc3));
