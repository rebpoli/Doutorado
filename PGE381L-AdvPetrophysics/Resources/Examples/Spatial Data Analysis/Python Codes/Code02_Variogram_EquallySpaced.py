# Variogram for equally-spaced data
# It is assumed that the x and y spacing are equal
# Modified from Gerry Middleton, November 1995.

import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt('Porosity.DAT')
h = 16 # Lag distance. You can change this parameter.

#  Plot Data
plt.figure(figsize=(10,7))
plt.pcolor(Data, edgecolors='k', linewidths=1)
plt.title('Porosity', fontsize=16, weight='bold')
plt.xlabel('X (km)', fontsize=16, weight='bold')
plt.ylabel('Y (km)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.colorbar()

#%% Calculate and plot variogram

# Calculate Varioram
r = len(Data[:,0])	#r is no of rows
c = len(Data[:,1]) # c no of cols
G = np.zeros((r,h))
for i in range(0,r): #for each row in turn
   for j in range(0,h):
      xx = Data[i,0:c-j-1] #data from 1st to (c-j)th col
      y = Data[i,j+1:c]  #data lagged by j to end of col
      G[i,j] = sum((xx-y)**2)/(2*(c-(j+1)))
      print(G[i,j])
gamax = sum(G)/r

for i in range(0,c): #for each row in turn
   for j in range(0,h):
      xx = Data[0:r-j-1,i] #data from 1st to (c-j)th col
      y = Data[j+1:r,i]  #data lagged by j to end of col
      G[i,j] = sum((xx-y)**2)/(2*(r-(j+1)))
      print(G[i,j])
gamay = sum(G)/c

# Plot Variogram
plt.figure(figsize=(10,7))
# Setup x values
x_divisions = np.linspace(0,h,h+1,True)
# Append 0 at the beginning of arrays
gamax0 = np.array([0])
gamax0 = np.append(gamax0,gamax)
gamay0 = np.array([0])
gamay0 = np.append(gamay0,gamay)
plt.scatter(x_divisions,gamax0,marker='o',facecolors='none',edgecolors=[0.1,0.1,1])
plt.plot(x_divisions,gamax0,':',label='X Direction')
plt.scatter(x_divisions,gamay0,marker='*',color=[0.1,0.5,0])
plt.plot(x_divisions,gamay0,':',label='Y Direction')
plt.xlabel('h (km)', fontsize=16, weight='bold')
plt.ylabel('$\gamma$(h)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')
plt.legend(prop={'size': 16})
plt.show()
