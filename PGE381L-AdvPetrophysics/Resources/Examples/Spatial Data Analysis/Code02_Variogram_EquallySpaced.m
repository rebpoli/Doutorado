% Variogram for equally-spaced data  
% It is assumed that the x and y spacing are equal
% Modified from Gerry Middleton, November 1995.

clear all;
close all;
clc;

Data = load('Porosity.dat');
h = 16; % Lag distance. You can change this parameter.

% Plot Data
figure;
grid on;box on;hold on
set(gcf,'color','white')
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);
surface(Data)
colorbar
xlabel('X (km)','fontsize',16,'fontweight','bold');
ylabel('Y (km)','fontsize',16,'fontweight','bold');
set(gca,'XGrid','on','YGrid','on','LineWidth',2);
title('Porosity');
%%
% Calculate Varioram
[r c] = size(Data);	%r is no of rows, c no of cols
for i = 1:r			%for each row in turn
   for j = 1:h
      xx = Data(i,1:c-j); %data from 1st to (c-j)th col
      y = Data(i,1+j:c);  %data lagged by j to end of col
      G(i,j) = sum((xx-y) .^ 2)/(2*(c-j));
      end
   end;
gamax = sum(G)/r;
for i = 1:c			%for each col in turn
   for j = 1:h
      xx = Data(1:r-j,i); %data from 1st to (r-j)th row
      y = Data(1+j:r,i);  %data lagged by j to end of row
      G(i,j) = sum((xx-y) .^ 2)/(2*(r-j)); %each row has gamma for a col
      end
   end;
gamay = sum(G)/c;	 %sum the cols and average

% Plot Variogram
figure;
grid on;box on;hold on
set(gcf,'color','white')
set(gca,'FontWeight','bold','FontSize',16,'LineWidth',2);
plot([0:h],[0 gamax],'o','color',[0.1 0.1 1],'Markersize',6)
plot([0:h], [0,gamay],'*','color',[0.1 0.5 0],'Markersize',6)
plot([0:h], [0 gamax],':')
plot([0:h], [0 gamay],':')
gam = (gamax + gamay)/2;
xlabel('h (km)','fontsize',16,'fontweight','bold');
ylabel('\gamma(h)','fontsize',16,'fontweight','bold');
legend('X Direction','Y Direction')
set(gca,'XGrid','on','YGrid','on','LineWidth',2);



