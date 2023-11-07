## This code calculates autocovarianve, autocorrelation, and variogram
# Example: Formations A, B, and C

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read file and load data to variable df
fileName = 'formationA.xlsx';
#fileName = 'formationB.xlsx';
#fileName = 'formationC.xlsx';

df = pd.read_excel(fileName, sheet_name='Sheet1')

# Print headings available in the file
print("Column headings:")
headings=df.columns

# Load data of the columns in different arrays.
Data = df[headings[1]].values
h = df[headings[0]].values

nLags = round(0.6*len(h));

Vario = np.zeros(nLags)
Cov = np.zeros(nLags)
end = len(Data)
for i in range(0,nLags):
   pato1 = Data[0:end-i]
   pato2 = Data[i:end]
   lenPato = len(Data[0:-(1+i)])
   Vario[i] = sum((Data[0:end-i]-Data[i:end])**2)/(2*len(Data[0:end-i]));
   Cov[i] = sum((Data[0:end-i] - np.mean(Data))*(Data[i:end]-np.mean(Data)))/(len(Data[0:end-i])-1);

#%% Plot Covarianve and Correlation
   
plt.figure(figsize=(10,7))
plt.subplot(3, 1, 1)
plt.stem(h,Data,linefmt='b-', markerfmt='bo');
plt.xlabel('h', fontsize=16, weight='bold')
plt.ylabel('Calcite Concentration', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')

plt.subplot(3, 1, 2)
plt.plot(h[0:nLags]-h[0],Cov,color=[0.1,0.1,1],linewidth=2)
plt.xlabel('h', fontsize=16, weight='bold')
plt.ylabel('Cov(h)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')

plt.subplot(3, 1, 3)
plt.plot(h[0:nLags]-h[0],Cov/Cov[0],color=[0.1,0.7,0.1],linewidth=2)
plt.xlabel('h', fontsize=16, weight='bold')
plt.ylabel('Corr(h)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')

plt.tight_layout()

#%% Plot Variogram

plt.figure(figsize=(10,7))
plt.subplot(3, 1, 1)
plt.stem(h,Data, linefmt='b-', markerfmt='bo')
plt.xlabel('h', fontsize=16, weight='bold')
plt.ylabel('Calcite Concentration', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')

plt.subplot(3, 1, 2)
plt.plot(h[0:nLags]-h[0],Cov,'b',linewidth=2)
plt.xlabel('h', fontsize=16, weight='bold')
plt.ylabel('Cov(h)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')

plt.subplot(3, 1, 3)
plt.plot(h[0:nLags]-h[0],Vario,'r',linewidth=2)
plt.plot(h[0:nLags],Cov[0]-Cov,'r--')
plt.xlabel('h', fontsize=16, weight='bold')
plt.ylabel('$\gamma$(h)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(zorder=0, linestyle='--')

plt.tight_layout()
plt.show()
