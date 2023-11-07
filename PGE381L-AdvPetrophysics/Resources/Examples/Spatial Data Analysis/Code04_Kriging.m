% Kriging  
% Modified from Matlab Recipes for earth sciences

clear all;
close all;
clc;

load TOC_Spatial.mat

% Change number of data points used for Kriging
x=x(1:end-40);
y=y(1:end-40);
z=z(1:end-40);

[X1,X2] = meshgrid(x);
[Y1,Y2] = meshgrid(y);
[Z1,Z2] = meshgrid(z);

H = sqrt((X1 - X2).^2 + (Y1 - Y2).^2);
B = 0.5*(Z1 - Z2).^2; 

%% Variogram Model
% you can change this model based on the variogram you estimated in
% "Code03_Variogram_UnequallySpaced" code

% Exponential model with nugget
nugget = 0;
C = .8;
range = 45;
Var_mod_A = (nugget + C*(1 - exp(-3*H/range))).*(H>0);

% % Linear model with nugget
% nugget = 0;
% slope = 0.03;
% Var_mod_A = (nugget + slope*H).*(H>0);

% % Spherical model with nugget
% nugget = 0;
% C = 0.8;
% range = 45;
% Var_mod_A = nugget + (C*(1.5*H/range-0.5*(H/...
%    range).^3).*(H<=range)+ C*(H>range));


%% Ordinary Kriging
n = length(Var_mod_A);
Var_mod_A(:,n+1) = 1;
Var_mod_A(n+1,:) = 1;
Var_mod_A(n+1,n+1) = 0;

Var_A_inv = inv(Var_mod_A);

Max_lim = ceil(max(max(x),max(y)));
R = 0:5:Max_lim;
[Xg1,Xg2] = meshgrid(R,R);

Xg=reshape(Xg1,[],1);
Yg=reshape(Xg2,[],1);

Z_K = Xg*NaN;
min_error_var = Xg*NaN;

for k = 1:length(Xg)
    K_vec = ((x - Xg(k)).^2+(y - Yg(k)).^2).^0.5;
    % exponential model
    Var_B_K = (nugget + C*(1 - exp(-3*K_vec/range))).*(K_vec>0);
    Var_B_K(n+1) = 1; 
    E = Var_A_inv*Var_B_K; 
    Z_K(k) = sum(E(1:n,1).*z); 
    min_error_var(k) = sum(E(1:n,1).*Var_B_K(1:n,1))+E(n+1,1); 
end

r = length(R);
Z_Kriging = reshape(Z_K,r,r);
min_error_var_Kriging = reshape(min_error_var,r,r);

%% Ploting

figure;
grid on;box on;hold on
set(gcf,'color','white')
set(gca,'FontWeight','bold','FontSize',12,'LineWidth',2);

subplot(1,3,1)
grid on;box on;hold on
set(gca,'FontWeight','bold','FontSize',12,'LineWidth',2);
scatter(x,y,100,z,'filled','MarkerEdgeColor','k')
colorbar('SouthOutside')
xlabel('X (km)','fontsize',12,'fontweight','bold');
ylabel('Y (km)','fontsize',12,'fontweight','bold');
set(gca,'XGrid','on','YGrid','on','LineWidth',2);
title('TOC (wt%)');

subplot(1,3,2)
grid on;box on;hold on
set(gca,'FontWeight','bold','FontSize',12,'LineWidth',2);
pcolor(Xg1,Xg2,Z_Kriging)
colorbar('SouthOutside')
xlabel('X (km)')
ylabel('Y (km)')
title('Kriging estimate')
set(gca,'XGrid','on','YGrid','on','LineWidth',2);
hold on
plot(x,y,'ok')


subplot(1,3,3)
grid on;box on;hold on
set(gca,'FontWeight','bold','FontSize',12,'LineWidth',2);
pcolor(Xg1,Xg2,min_error_var_Kriging)
title('Kriging Minimum Error Variance')
xlabel('X (km)')
ylabel('Y (km)')
colorbar('SouthOutside')
plot(x,y,'ok')
