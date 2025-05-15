close all
clear
clc

area =55;   % ���쳣��Ϊ15���ṹ�쳣��Ϊ55
load 'D:\ͼ����\����-ר��-����\DFAE��������\HSIs\sandiego1'  % �õ�map�����ں������AUCʱ��ʹ��
load 'D:\ͼ����\����-ר��-����\DFAE��������\data_low_res\sandiego1_low'
load 'D:\ͼ����\����-ר��-����\DFAE��������\data_high_res\sandiego1_high'
savepath_res = 'D:\ͼ����\����-ר��-����\DFAE��������\data_res\sandiego1';

tic
%---------------------------��Ƶ�������˲����-----------------------------
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
% imshow(mat2gray(output1))   %���Ա任��ʾͼ��
% axis off,axis image

%-----------------------------�����˲�ȥ��---------------------------------
output2 = mat2gray(output1);
r_h = 1;   
eps_h = 0.12^2;   %0.06 urban1ѵ��   0.03
output3 = guidedfilter(output2, output2, r_h, eps_h);

r_h = 1;   
eps_h = 0.2^2;    % 2
output_h = guidedfilter(high, high, r_h, eps_h);
%--------------------------ָ���ں�-----------------------------
D1 = mat2gray(output_h);
D0 = mat2gray(output3);
% D = D1.*D0;

a = 1;
exp_ori = -a*D1;
k = 1 - exp(exp_ori);
D = k.*D0;
toc

% ��������
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
% %     axis([1,20,0.95,1])  %ȷ��x����y���ͼ��С
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
xlabel('��Ƶ��')
subplot(1,4,2)
imshow(output5)
xlabel('��Ƶ��')
subplot(1,4,3)
imshow(p3)
xlabel('�ںϽ��')
subplot(1,4,4)
imshow(map)
xlabel('map')

% figure()
% imshow(high1)
% figure()
% imshow(output5)
figure()
imshow(p3)
% ----------------------------- draw ROC���� ----------------------------------------
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

%---------------------������---------------------------------
DFAE = D2;
save(savepath_res,'DFAE');

%---------------------draw ROC�龯---------------------------------
[row,col,bands]=size(data);
N=row * col; %N�Ǹ߹���ͼ����г��У������ظ���
targets = reshape(map, 1, N); %map��ground truth

outputs3 = reshape(D0, 1, N);%res�Ǽ����
[FPR0,TPR0] = myPlotROC(targets, outputs3);
x = linspace(1e-4,1,1003);
auc_low = trapz(x,FPR0);     %�����ݶ��������
figure,plot(x,FPR0);
xlabel('false alarm rate');
ylabel('probability of detection');
title('ROC curves of detection algorithm');
text(0.5,0.7,sprintf('AUCROC_ =%f',auc_low));

outputs4 = reshape(D1, 1, N);
[FPR1,TPR1] = myPlotROC(targets, outputs4);
x = linspace(1e-4,1,1003);
auc_high = trapz(x,FPR1);     %�����ݶ��������
figure,plot(x,FPR1);
xlabel('false alarm rate');
ylabel('probability of detection');
title('ROC curves of detection algorithm');
text(0.5,0.7,sprintf('AUCROC_ =%f',auc_high));

outputs1 = reshape(mat2gray(D), 1, N);   %�ںϽ��
[FPR,TPR] = myPlotROC(targets, outputs1);
x = linspace(1e-4,1,1003);
auc3 = trapz(x,FPR);     %�����ݶ��������
figure,plot(x,FPR);
xlabel('false alarm rate');
ylabel('probability of detection');
title('ROC curves of detection algorithm');
text(0.5,0.7,sprintf('AUCROC_ =%f',auc3));
