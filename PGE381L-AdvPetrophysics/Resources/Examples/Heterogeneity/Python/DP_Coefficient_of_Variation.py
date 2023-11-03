
# import library pandas to import ExcelFile
import pandas as pd

# import library numpy to optimize array calculations performance
import numpy as np

# import library matplotlib to plot data
import matplotlib.pyplot as plt
import matplotlib as mpl

#import function to find the Quantile
from QuantileFunc import QuantileFunc
from ECDFFunc import ECDFFunc
from ExponentialFunc import ExponentialFunc

import math

# import library matplotlib to calculate the MODE of a Dataset
from scipy import special
from scipy.stats import norm
from scipy.optimize import curve_fit


# Read file and load data to variable df
fileName = 'Carbonate1 Data.xlsx'
#fileName = 'Carbonate2 Data.xlsx'
coreM = pd.read_excel(fileName, sheet_name='Sheet1')

# Print headings available in the file
print("Column headings:")
print(coreM.columns)

# Load data of the columns in different arrays. To do this you only need the name of the heading
Depth = coreM['Depth (ft) ']
Perm = coreM['Permeability (md)']

#Calculate the experimental cumulative distribution function (ECDF) of a dataset. 
# Call the function ECDF(Dataset) that returns the sorted data and the cumulative cdf values
# Into the (), imput your Dataset.
(cum_perm,eval_perm) = ECDFFunc(Perm)

#Sort the permeability in descending order
perm_sort = sorted(Perm,reverse=True)

#Calculate the quantiles or percentiles
# Call the function QuantileFunc(Dataset,Quantile number from 0 to 1) to calculate the quantile of the dataset
# Into the (), imput your Dataset and the number of quantile in fraction, separated by a ','.
Q1 = QuantileFunc(perm_sort,(1-0.841))
Q2 = QuantileFunc(perm_sort,0.5)

Ks = [Q1,Q2]

K50 = Ks[1]
K84 = Ks[0]

#Dykstra-Parson's Coefficient
V = (K50-K84)/K50
print('The heterogeneity coefficient is {:.4f}'.format(V))

tol = 1e-10;
#x[0] = x[0] + tol;
#x[-1] = x[-1] - tol;
cum_perm[-1] = 0.99;

#Probit function
probit = math.sqrt(2) * special.erfinv(2*cum_perm - 1)
probit2 = norm.ppf(cum_perm)

# change figure size
plt.figure(figsize=(6,5))

plt.subplot(1, 2, 1)
x = probit[1:-1]
y = eval_perm[1:-1]
popt, pcov = curve_fit(ExponentialFunc, x, y, p0=[2000., 0.005])
plt.plot(x,y,'ko', ms=3)
xmin = min(probit[1:-1])
xmax = max(probit[1:-1])
xx = np.linspace(xmin, xmax, 1000)
yy = ExponentialFunc(xx, *popt)
plt.plot(xx,yy,'r',linewidth=2)
# Modify the axis margens
plt.margins(x=0)
# Name the plot axis labels and title
plt.xlabel('x', fontsize=16, weight='bold')
plt.ylabel('y', fontsize=16, weight='bold')
# Define axis range
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
# Define the direction of the axis ticks.
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
plt.grid(zorder=0, linestyle='--')
plt.yscale('log')

plt.subplot(1, 2, 2)
x = probit2[1:-1]
y = eval_perm[1:-1]
popt, pcov = curve_fit(ExponentialFunc, x, y, p0=[2000., 0.005])
plt.plot(x,y,'ko', ms=3)
xmin = min(probit2[1:-1])
xmax = max(probit2[1:-1])
xx = np.linspace(xmin, xmax, 1000)
yy = ExponentialFunc(xx, *popt)
plt.plot(xx,yy,'r',linewidth=2)
# Modify the axis margens
plt.margins(x=0)
# Name the plot axis labels and title
plt.xlabel('x', fontsize=16, weight='bold')
plt.ylabel('y', fontsize=16, weight='bold')
# Define axis range
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
# Define the direction of the axis ticks.
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
plt.grid(zorder=0, linestyle='--')
plt.yscale('log')

plt.tight_layout()
plt.show()