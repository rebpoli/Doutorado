%% This code calculates autocovarianve, autocorrelation, and variogram
% Example: Formations A, B, and C 

close all
clear all
clc

File = xlsread('ForA.xlsx');
% File = xlsread('ForB.xlsx');
% File = xlsread('ForC.xlsx');

Data = File(:,2);
h = File(:,1);

nLags = round(0.6*length(h));

for i = 0:nLags-1,
    Vario(i+1) = sum((Data(1:end-i)-Data(1+i:end)).^2)/(2*length(Data(1:end-i)));
    Cov(i+1) = sum((Data(1:end-i) - mean(Data)).*(Data(1+i:end)-mean(Data)))/(length(Data(1:end-i))-1);
end


%% Plot Covarianve and Correlation
figure;
grid on;box on;hold on
set(gcf,'color','white')

subplot(3,1,1)
stem(h,Data)
xlabel('h','fontsize',16,'fontweight','bold');
ylabel('Calcite Concentration','fontsize',16,'fontweight','bold');
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);

subplot(3,1,2)
plot(h(1:nLags)-h(1),Cov,'color',[0.1 0.1 1],'Linewidth',2)
xlabel('h','fontsize',16,'fontweight','bold');
ylabel('Cov(h)','fontsize',16,'fontweight','bold');
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);

subplot(3,1,3)
plot(h(1:nLags)-h(1),Cov/Cov(1),'color',[0.1 0.7 0.1],'Linewidth',2)
xlabel('h','fontsize',16,'fontweight','bold');
ylabel('Corr(h)','fontsize',16,'fontweight','bold');
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);

%% Plot Variogram
figure;
grid on;box on;hold on
set(gcf,'color','white')

subplot(3,1,1)
stem(h,Data)
xlabel('h','fontsize',16,'fontweight','bold');
ylabel('Calcite Concentration','fontsize',16,'fontweight','bold');
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);

subplot(3,1,2)
plot(h(1:nLags)-h(1),Cov,'color',[0.1 0.1 1],'Linewidth',2)
xlabel('h','fontsize',16,'fontweight','bold');
ylabel('Cov(h)','fontsize',16,'fontweight','bold');
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);

subplot(3,1,3)
plot(h(1:nLags)-h(1),Vario,'color',[1 0.1 0.1],'Linewidth',2)
hold on
plot(h(1:nLags),Cov(1)-Cov,'r--')
xlabel('h','fontsize',16,'fontweight','bold');
ylabel('\gamma(h)','fontsize',16,'fontweight','bold');
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);
