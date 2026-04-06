# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:18:03 2022

@author: TS
"""

import pandas as pd
import numpy as np
import scipy.stats
import pip
pip.main(['install','seaborn'])
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots
plt.style.use('fivethirtyeight') # For plots

# To list all the available styles for plotting:
from matplotlib import style
print(plt.style.available)
plt.style.use('YOUR DESIRED STYLE')

#Checking the files in the working directory
import os
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

 ## Open the 2021 dataset text file
 
# Provide the path to your text file
file_path = "/Users/taraneh/Documents/PhD Thesis/Thesis Data/T-Dinamik Konya SPP/Energy Data/2021_DayEnergyValues.txt"
# Read the dataset into a pandas DataFrame
df = pd.read_csv(file_path, delimiter='\t')
# Optionally, you can display the first few rows to inspect the data
print(df.head())
del df['Unnamed: 9']

# Convert the 'Time' column to pandas datetime object
df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S')
#Making Time column as index
df.set_index('Time', inplace = True)

# Calculation for Energy values at each specific Time
df_diffed = df.diff()
df_diffed = df_diffed.dropna()
'''row_indexer= (df_diffed.iloc[:, 0:] < 0).any(axis='columns')
df_diffed = df_diffed.drop(df_diffed[row_indexer].index)'''

# Saving in excel file 
file_name = 'Differenced Data 2021 NEW.xlsx'
df_diffed.to_excel(file_name)
# Reading from excel
xls = pd.ExcelFile('Differenced Data 2021 NEW.xlsx')
df_diffed_2021 = pd.read_excel(xls)
df_diffed_2021.set_index('Time', inplace = True)


# Opening clean data (outlier removed) from Excel
xls = pd.ExcelFile('***REVISED_LOF_Clean_final_twoyears_data_Rescaled.xlsx')
df_final = pd.read_excel(xls)
del df_final['Unnamed: 0']
df_final.set_index('Time', inplace = True)
# Data Visualization
df_final[df_final < 0] = 0
df_final.plot(linewidth = 0.5).legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)
 ### End of LOF ###

# Creating the Inverters Data Frame
"""df_inverters = dataframe.iloc[:,0:8]"""
# Showing each inverter production seperateley
df_final.plot(subplots=True, figsize=(8, 8))

# Comparing the Maximum values of inverters
df_max = df_final.iloc[:,0:8].max(axis=0)

# Comparing minimum values
df_min = df_final.iloc[:,0:8].min(axis=0)


# Creating Season Data
season_data = df_final.resample("Q").sum()
# Season Data Plot
season_data.iloc[:,0:8].plot()
# Creating Monthly Data
monthly_data = df_final.resample("M").sum()
# Monthly Plot
monthly_data.iloc[:,0:8].plot().legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)
# Creating Daily Data
daily_data = df_final.resample("D").sum()
# Daily Plot
daily_data.iloc[:,0:8].plot(linewidth=1.0).legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=2)
# Creating Hourly Data
hourly_data = df_final.resample("H").sum()
# Hourly Plot
hourly_data.iloc[:,0:8].plot(linewidth=0.5).legend(loc=2, prop={'size': 6})


# EXPLORATORY DATA ANALYSIS

# Decomposing Time Series Data
decompose_result_add = seasonal_decompose(daily_data['INV/5/DayEnergy (kWh)'], model="additive")
decompose_result_add.plot().linewidth = 0.2
trend = decompose_result_add.trend
seasonal = decompose_result_add.seasonal
residual = decompose_result_add.resid

trend.plot(linewidth = 1.0)
seasonal.plot(linewidth=0.5, color='r')
residual.plot(linewidth=0.5, color = 'orange')
plt.legend(loc='upper center', bbox_to_anchor=(0.35, -0.35), fancybox=True, shadow=True, ncol=3)

#Fitting a line(regression)
x = np.arange(daily_data.index.size)
fit = np.polyfit(x, daily_data['INV/5/DayEnergy (kWh)'], deg=50)
#Fit function : y = mx + c [linear regression ]
fit_function = np.poly1d(fit)

#Linear regression plot
plt.plot(daily_data.index, fit_function(x))
#Time series data plot
plt.plot(daily_data.index, daily_data['INV/5/DayEnergy (kWh)'],linewidth=0.5)
plt.xlabel('Date')
plt.ylabel('INV/5/DayEnergy (kWh)')
plt.title('Polynomial regression degree:50')
plt.show()
# model evaluation
rmse = mean_squared_error(daily_data['INV/5/DayEnergy (kWh)'], fit_function(x))
r2 = r2_score(daily_data['INV/5/DayEnergy (kWh)'], fit_function(x))


# Auto Correlation for each inverter(columns)
df_corr = df_final[df_final.columns.to_list()].apply(lambda x: x.autocorr())

# Creating Autocorrelation plot
pd.plotting.autocorrelation_plot(df_final['INV/5/DayEnergy (kWh)'], linewidth = 0.5).grid(True)

# Focused area AC for 1000 data 
pd.plotting.autocorrelation_plot(df_final.iloc[0:1000,4], ax = None, linewidth = 0.5).grid(True)

# Signal Processing
# PSD
plt.psd(df_final['INV/5/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5)
plt.ylabel('Power Spectral Density (W^2*S^3)')

# CPSD
plt.csd(df_final['INV/5/DayEnergy (kWh)'], df_final['INV/6/DayEnergy (kWh)'], Fs=1/86.400, NFFT=1024, linewidth = 1.5)
plt.ylabel(' Cross Power Spectral Density (W^2*S^3)')

# Cross Correlation
from scipy import signal
def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))  
    c = np.correlate(p, q, 'full')
    return c
    
ccf_inverters = ccf_values(df_final['INV/5/DayEnergy (kWh)'].iloc[0:1000], df_final['INV/6/DayEnergy (kWh)'].iloc[0:1000])
ccf_inverters

# Create a list of of our lag values and visualize it against the correlation coefficients
lags = signal.correlation_lags(len(df_final['INV/5/DayEnergy (kWh)'].iloc[0:1000]), len(df_final['INV/6/DayEnergy (kWh)'].iloc[0:1000]))

# Plot CCF
def ccf_plot(lags, ccf):
    fig, ax =plt.subplots(figsize=(9, 6))
    ax.plot(lags, ccf, linewidth = 0.5)
    ax.axhline(-2/np.sqrt(23), color='red', label='5% confidence interval')
    ax.axhline(2/np.sqrt(23), color='red')
    ax.axvline(x = 0, color = 'black', lw = 1)
    ax.axhline(y = 0, color = 'black', lw = 1)
    ax.axhline(y = np.max(ccf), color = 'blue', lw = 1, linestyle='--', label = 'highest +/- correlation')
    ax.axhline(y = np.min(ccf), color = 'blue', lw = 1, linestyle='--')
    ax.set(ylim = [-1, 1])
    ax.set_title('Cross Correation of Two Inverters in a week', weight='bold', fontsize = 15)
    ax.set_ylabel('Correlation Coefficients', weight='bold', fontsize = 12)
    ax.set_xlabel('Time Lags', weight='bold', fontsize = 12)
    plt.legend()
    
ccf_plot(lags, ccf_inverters)

# Coherence
plt.cohere(df_final['INV/5/DayEnergy (kWh)'], df_final['INV/6/DayEnergy (kWh)'], Fs=1/86.400, NFFT=1024, linewidth = 1.5)

# PSD for All inverters in one graph
plt.psd(df_final['INV/1/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.,label='INV_1')
plt.psd(df_final['INV/2/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5,label='INV_2')
plt.psd(df_final['INV/3/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5,label='INV_3')
plt.psd(df_final['INV/4/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5,label='INV_4')
plt.psd(df_final['INV/5/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5,label='INV_5')
plt.psd(df_final['INV/6/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5,label='INV_6')
plt.psd(df_final['INV/7/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5,label='INV_7')
plt.psd(df_final['INV/8/DayEnergy (kWh)'], NFFT=1024, Fs=1/86.400, linewidth=1.5,label='INV_8')
plt.ylabel('Power Spectral Density (W^2*S^3)')
plt.legend(loc = "lower center", bbox_to_anchor=(0.45, -0.45), ncol=4)

# Stationarity Test 
from statsmodels.tsa.stattools import adfuller

dftest = adfuller(df_final['INV/5/DayEnergy (kWh)'])
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput) 

# Make stationary 
df_diff = daily_data['K1SonTedarik_Business_AG'].diff(1).dropna()
df_diff = pd.DataFrame(df_diff)
dftest = adfuller(df_diff['K1SonTedarik_Business_AG'])
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)
























