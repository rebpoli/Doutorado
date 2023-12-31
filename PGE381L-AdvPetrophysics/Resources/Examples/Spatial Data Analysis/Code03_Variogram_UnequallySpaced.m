% Variogram Analysis 
% Modified from Matlab Recipes for earth sciences 

clear all;
close all;
clc;

load TOC_Spatial.mat

%% Data Ploting and Initial Analysis
figure;
grid on;box on;hold on
set(gcf,'color','white')
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);
scatter(x,y,100,z,'filled','MarkerEdgeColor','k')
colorbar
xlabel('X (km)','fontsize',16,'fontweight','bold');
ylabel('Y (km)','fontsize',16,'fontweight','bold');
set(gca,'XGrid','on','YGrid','on','LineWidth',2);
title('TOC (wt%)');

[X1,X2] = meshgrid(x);
[Y1,Y2] = meshgrid(y);
[Z1,Z2] = meshgrid(z);

H = sqrt((X1 - X2).^2 + (Y1 - Y2).^2);
B = 0.5*(Z1 - Z2).^2; 

%% Variogram Calculations

H2 = H.*(diag(x*NaN)+1);
lag = mean(min(H2))
hmd = max(H(:))/2
max_lags = floor(hmd/lag)
LAGS = ceil(H/lag);

for i = 1:max_lags
    SEL = (LAGS == i);
    H_Var(i) = mean(mean(H(SEL)));
    B_Var(i) = mean(mean(B(SEL)));
end

Var_z = var(z) 
b = [0 max(H_Var)]; 
c = [Var_z Var_z];

figure;
grid on;box on;hold on
set(gcf,'color','white')
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);
plot(H_Var,B_Var,'.','color',[0.1 0.1 1],'Markersize',16)
plot(b,c, '--r') 
ylim([0 1.21*max(B_Var)])
xlabel('Lag distance (km)','fontsize',16,'fontweight','bold');
ylabel('Variogram','fontsize',16,'fontweight','bold');
set(gca,'XGrid','on','YGrid','on','LineWidth',2);

%% Analytical Models
lags=0:max(H_Var);

%% Spherical model with nugget
% % Change the constants to get a reasonable match
% nugget = 0;
% C = 0.8;
% range = 45;
% Bsph = nugget + (C*(1.5*lags/range-0.5*(lags/...
%    range).^3).*(lags<=range)+ C*(lags>range));
% plot(lags,Bsph,'-','color',[1 0 0],'Linewidth',2)

 
%% Exponential model with nugget
% % Change the constants to get a reasonable match
% 
% nugget = 0.1;
% C = 0.7;
% range = 45;
% Bexp = nugget + C*(1 - exp(-3*lags/range));
% plot(lags,Bexp,'-','color',[0.3 0.7 0.8],'Linewidth',2)
%  
%% Linear model with nugget
% Change the constants to get a reasonable match

% nugget = 0;
% slope = 0.03;
% Blin = nugget + slope*lags;
% plot(lags,Blin,'-','color',[0.1 0.6 0.1],'Linewidth',2)
% hold off

