clear all;  clc; 
close all

addpath(genpath('../Data'))

%name = 'abu-beach-3';
%Name = {'abu-beach-2', 'abu-beach-3','abu-urban-4','abu-urban-5','Segundo','sandiego_plane','GrandIsle'};
Name = {'abu-beach-1','HYDICE_data','sandiego_plane', 'abu-beach-2', 'abu-urban-3'};
for i=[1:5]
    %i = 6
    name = char(Name(i))
    %name = 'Segundo'
    %name = 'abu-airport-3';
    %name = 'HYDICE_data';
    %name = 'sandiego_data';
    %name = 'sandiego_plane';
    load (['D:/avirs数据/abu/',name,'.mat'])
    [col,row] = size(map);
    symbol = {'-*','-O','-^','-s','-p','-+','-x'};
    figure('color',[1,1,1]),
    hold on
    %% RX
    load(['./RX',name,'.mat'])
    r0 = RX;
    r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
    [PD0 PF0] = roc(map(:), r0(:));
    plot(PF0,PD0,'--','LineWidth',1.5)
    areaRX = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);

    %% LRX
    load(['./LRX',name,'.mat'])
    r0 = LRX;
    r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
    [PD0 PF0] = roc(map(:), r0(:));
    plot(PF0,PD0,'-.','LineWidth',1.5)
    areaLRX = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
    
    %% CRD
    load(['./CRD',name,'.mat'])
    r0 = CRD;
    r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
    [PD0 PF0] = roc(map(:), r0(:));
    plot(PF0,PD0,':','LineWidth',1.5)
    areaCRD = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);

    %% LRSAR
    load(['./LRASR',name,'.mat'])
    r0 = array;
    r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
    [PD0 PF0] = roc(map(:), r0(:));
    plot(PF0([1:int64(length(PF0)*200)/10000:length(PF0)]),PD0([1:int64(length(PF0)*200)/10000:length(PD0)]),'-+','LineWidth',1.5)
    areaLRSAR = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
    %% LSMAD
    load(['./LSMAD',name,'.mat'])
    r0 = array;
    r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
    [PD0 PF0] = roc(map(:), r0(:));
    plot(PF0([1:int64(length(PF0)*200)/10000:length(PF0)]),PD0([1:int64(length(PF0)*200)/10000:length(PD0)]),'-p','LineWidth',1.5)
    areaLSMAD = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
        %% AED
    load(['./AED',name,'.mat'])
    r0 = AED;
    r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
    [PD0 PF0] = roc(map(:), r0(:));
    plot(PF0([1:int64(length(PF0)*50)/10000:length(PF0)]),PD0([1:int64(length(PF0)*50)/10000:length(PD0)]),'-s','LineWidth',1.5)
    areaAED = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
    %% AE-GMM
%     load(['E:\论文写作\dagmm\GMM_based_comparsion_algorithm\AE_GMM_cp\AE_GMM_cp\AE_GMM',name,'.mat'])
%     r0 = array;
%     r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
%     [PD0 PF0] = roc(map(:), r0(:));
%     plot(PF0([1:1:length(PF0)]),PD0([1:1:length(PD0)]),'-','LineWidth',3)
%     areaLAEGMM = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
    %% Proposed method
    %load(['E:\论文写作\dagmm\hidden_nodes=8_with2rec_4connection_postprocessing_with_pca\',name,'\',name,'Energy-map-cluster centers5hidden nodes8_Gamma_transform_Gamma5.mat'])
    %load(['E:\论文写作\dagmm\hidden_nodes=8_with2rec_4connection_postprocessing_with_pca\',name,'\',name,'Energy-map-cluster centers=5hidden nodes=8.mat'])
    load(['J:\论文写作\dagmm\graph regulized\gaelambda_adjusthidden_nodes=8_with2rec_4connection_postprocessing_with_pca_1000epoch\',name,'\',name,'Post-process-Energy-map-cluster centers=5hidden nodes=8l1=0.1l2=0.001l3=3eps=0.01.mat'])
    r0 = reshape(array,col,row);
    r0 = mat2gray(r0);
%     figure,imagesc(r0)
%     axis equal
%     axis off
    [PD0 PF0] = roc(map(:), r0(:));
    plot(PF0,PD0,'-','LineWidth',3)  
    areaPM = -sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
    box on
    xlabel('FPR','Fontname','Times New Roman','FontSize',10,'FontWeight','bold')
    ylabel('TPR','Fontname','Times New Roman','FontSize',10,'FontWeight','bold')
    %legend({'RX','LRX','CRD','LRASR','LSMAD', 'AED','Proposed method'},'Fontname', 'Times New Roman','FontWeight','bold','FontSize',10,'location','southeast');
     set(gca,'xscale','log')
%     set(gca,'yscale','log')
    set(gca,'Fontname','Times New Roman','FontWeight','bold','Linewidth',1.5)
    axis([1e-4 1 0 1])
    set(gcf,'position',[200,200,260,215])
    %title(name)
end
