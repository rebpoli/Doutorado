
# import library pandas to import ExcelFile
import pandas as pd

# import library numpy to optimize array calculations performance
import numpy as np

# import library statistics to calculate the Standard deviation of a dataset
import statistics


# Read file and load data to variable df
fileName = 'Carbonate1 Data.xlsx'
#fileName = 'Sandstone1 Data.xlsx'
coreM = pd.read_excel(fileName, sheet_name='Sheet1')

# Print headings available in the file
print("Column headings:")
print(coreM.columns)

# Load data of the columns in different arrays. To do this you only need the name of the heading
Perm = coreM['Permeability (md)']

## Calculate the coefficient of variation

MEAN = np.mean(Perm)
STD = statistics.stdev(Perm)

CV = STD / MEAN

print(MEAN)
print(STD)
print(CV)

