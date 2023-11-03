## Kriging
# Modified from Matlab Recipes for earth sciences

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_excel('TOC_Spatial.xlsx', sheet_name='Sheet1')

# Print headings available in the file
print("Column headings:")
headings=Data.columns

x = Data[headings[0]].values
y = Data[headings[1]].values
z = Data[headings[2]].values

X1,X2 = np.meshgrid(x,x)
Y1,Y2 = np.meshgrid(y,y)
Z1,Z2 = np.meshgrid(z,z)

H = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
B = 0.5*(Z1 - Z2)**2

#%% Variogram Model
# you can change this model based on the variogram you estimated in
# "Code03_Variogram_UnequallySpaced" code

# %% Exponential model with nugget
# Change the constants to get a reasonable match
nugget = 0
C = 0.8
#range = 45; #range is a protected function in python
rng = 45 # Range 
Var_mod_A = (nugget + C*(1 - np.exp(-3*H/rng)))*(H>0)

#
# %% Linear model with nugget
# nugget = 0
# slope = 0.03
# Var_mod_A = (nugget + slope*H)*(H>0)

# %% Spherical model with nugget
# Change the constants to get a reasonable match
# nugget = 0
# C = 0.8
## rng = 45 #range is a protected function in python
#rng = 45 #Range
# Var_mod_A = nugget + (C*(1.5*H/rng-0.5*(H/range)**3)*(H<=rng)+ C*(H>rng))

#%% Ordinary Kriging
n = len(Var_mod_A)
#add a column of ones
Var_mod_A = np.c_[ Var_mod_A, np.ones(n) ]
n1=len(Var_mod_A)
#add a row of ones
Var_mod_A = np.r_[ Var_mod_A, [np.ones(n+1).T]]
#element (n,n)=0
Var_mod_A[n,n] = 0

Var_A_inv = np.linalg.inv(Var_mod_A)

Max_lim = np.ceil(np.max([np.max(x),np.max(y)]))
divisions = Max_lim / 5
R = np.linspace(0,Max_lim,divisions+1,True)
Xg1,Xg2 = np.meshgrid(R,R)

# All matrix values in vectors
Xg=Xg1.flatten('F')
Yg=Xg2.flatten('F')

Z_K = Xg*float('nan')
min_error_var = Xg*float('nan')

lenXg = len(Xg)

for k in range(0,len(Xg)):
    K_vec = ((x - Xg[k])**2+(y - Yg[k])**2)**0.5
    # exponential model
    #range is a protected function in python, changed to rng
    Var_B_K = (nugget + C*(1 - np.exp(-3*K_vec/rng)))*(K_vec>0)
    # add 1 at the end of the vector
    Var_B_K = np.r_[ Var_B_K, [1]]
    E = np.matmul(Var_A_inv,Var_B_K)
    Z_K[k] = sum(E[0:n]*z)
    min_error_var[k] = sum(E[0:n]*Var_B_K[0:n])+E[n]
    pato=0


r = len(R)
Z_Kriging = Z_K.reshape(r,r,order='F')
min_error_var_Kriging = min_error_var.reshape(r,r,order='F')

# Change plot size
plt.figure(figsize=(10,7))
plt.subplot(1, 3, 1)
# Plot Data
plt.scatter(x,y,c=z,s=300,edgecolor='black')
# Name the plot axis labels and title
plt.title('TOC (wt%)', fontsize=16, weight='bold')
plt.xlabel('X (km)', fontsize=16, weight='bold')
plt.ylabel('Y (km)', fontsize=16, weight='bold')
plt.xlim(0,np.ceil(np.max(x)))
plt.ylim(0,np.ceil(np.max(y)))
# Define axis range
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.colorbar(orientation='horizontal')
plt.grid(zorder=0, linestyle='--')

plt.subplot(1, 3, 2)
# Plot Data
plt.pcolor(Xg1,Xg2,Z_Kriging, edgecolors='k', linewidths=0.4)
plt.plot(x,y,'o',color=[0,0,0],fillstyle='none')
# Name the plot axis labels and title
plt.title('Kriging estimate', fontsize=16, weight='bold')
plt.xlabel('X (km)', fontsize=16, weight='bold')
plt.ylabel('Y (km)', fontsize=16, weight='bold')
# Define axis range
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.colorbar(orientation='horizontal')

plt.subplot(1, 3, 3)
# Plot Data
plt.pcolor(Xg1,Xg2,min_error_var_Kriging, edgecolors='k', linewidths=0.4)
plt.plot(x,y,'o',color=[0,0,0],fillstyle='none')
# Name the plot axis labels and title
plt.title('Kriging Minimum Error Variance', fontsize=16, weight='bold')
plt.xlabel('X (km)', fontsize=16, weight='bold')
plt.ylabel('Y (km)', fontsize=16, weight='bold')
# Define axis range
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.colorbar(orientation='horizontal')

plt.tight_layout()

plt.show()
