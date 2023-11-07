clear all 
close all
clc

% filename = 'Carbonate1 Data.xlsx';
filename = 'Sandstone1 Data.xlsx';

coreM = xlsread(filename);
perm = coreM(:,3);

Cv = std(perm)/mean(perm)

