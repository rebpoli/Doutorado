clear all 
close all
clc

filename = 'Carbonate1 Data.xlsx';
% filename = 'Carbonate2 Data.xlsx';

coreM = xlsread(filename);

% perm = coreM(:,1);
perm = coreM(:,3);

% Calculate the cumulative distribution of the permeability
[cum_perm,eval_perm] = ecdf(perm);

% Sort the permeability in descending order
perm_sort = sort(perm,'descend');

% Calculate the percentiles
Ks = quantile(sort(perm,'descend'),[(1-0.841) 0.50]); 

K50 = Ks(2); % median
K84 = Ks(1);

% Dykstra-Parson's Coefficient
V = (K50-K84)/K50;

fprintf('The heterogeneity coefficient is %.4f.\n',V);

tol = 1e-10;
%cum_perm(1) = cum_perm(1) + tol;
%cum_perm(end) = cum_perm(end) - tol;
cum_perm(end) = 0.99;

% Probit function
probit = sqrt(2)*erfinv(2*cum_perm - 1);
probit2 = norminv(cum_perm);

subplot(1,2,1)
f = fit(probit(2:end-1),eval_perm(2:end-1),'exp1')
plot(f,probit(2:end-1),eval_perm(2:end-1));
set(gca,'yscale','log')
grid on

subplot(1,2,2)
f = fit(probit2(2:end-1),eval_perm(2:end-1),'exp1')
plot(f,probit2(2:end-1),eval_perm(2:end-1));
set(gca,'yscale','log')
grid on
