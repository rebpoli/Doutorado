# Variogram Analysis
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

h = 16 # Lag distance. You can change this parameter.

#%% Data Ploting and Initial Analysis

# Change plot size
plt.figure(figsize=(10,7))
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
plt.colorbar()
plt.grid(zorder=0, linestyle='--')

X1,X2 = np.meshgrid(x,x)
Y1,Y2 = np.meshgrid(y,y)
Z1,Z2 = np.meshgrid(z,z)

H = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
B = 0.5*(Z1 - Z2)**2

#%% Variogram Calculations

H2 = np.zeros((len(H),len(H)))
H2 = H

arrayMin = np.zeros(len(H))
for i in range(0,len(H)):
    H2[i,i] = float('NaN')
    arrayMin[i] = np.nanmin(H2[i,:])

lag = np.mean(arrayMin)
hmd = np.nanmax(H)/2
max_lags = int(np.floor(hmd/lag))
LAGS = np.ceil(H/lag)

H_Var = np.zeros(max_lags)
B_Var = np.zeros(max_lags)

for i in range(0,max_lags):
    SEL = LAGS == i+1
    H_Var[i] = np.mean(np.nanmean(H[SEL]))
    B_Var[i] = np.mean(np.mean(B[SEL]))

Var_z = np.var(z, ddof=1)
b = np.array([0,max(H_Var)])
c = np.array([Var_z, Var_z])

# Change plot size
plt.figure(figsize=(10,7))
# Plot Data
plt.scatter(H_Var,B_Var)
plt.plot(b,c,'r--')
# Name the plot axis labels and title
plt.xlabel('Lag distance (km)', fontsize=16, weight='bold')
plt.ylabel('Variogram', fontsize=16, weight='bold')
# Define axis range
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')

# %% Analytical Models

maxH_Var = int(np.max(H_Var))
lags = np.linspace(0,maxH_Var,maxH_Var+1,True)

# %% Spherical model with nugget
## Change the constants to get a reasonable match

nugget = 0
C = 0.8
rng = 45 # Range
Bsph = nugget + (C*(1.5*lags/rng-0.5*np.power((lags/rng),3))*(lags<=rng)+ C*(lags>rng));
plt.plot(lags,Bsph,'-',color=[1,0,0],linewidth=2)

# %% Exponential model with nugget
## Change the constants to get a reasonable match

#nugget = 0
#C = 0.8
#rng = 45 # Range
#Bexp = nugget + C * (1 - np.exp(-3 * lags / rng))
#plt.plot(lags, Bexp, '-', color=[0.3,0.7,0.8], linewidth=2)
#plt.xlim(0,maxH_Var+1)


# %% Linear model with nugget
## Change the constants to get a reasonable match

#nugget = 0
#slope = 0.03
#Blin = nugget + slope*lags
#plt.plot(lags,Blin,'-',color=[0.1,0.6,0.1],linewidth=2)

plt.show()
