#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:14:49 2024

@author: taraneh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 13:01:23 2023

@author: taraneh
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
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight') # For plots



# Removing outliers with LOF

   # Rescaling 2022 production value data
# Reading differenced data from excel
xls = pd.ExcelFile('Differenced Data 2021 NEW.xlsx')
df_diffed_2021 = pd.read_excel(xls)
df_diffed_2021['Time'] = pd.to_datetime(df_diffed_2021['Time'], format='%d/%m/%Y %H:%M:%S')
df_diffed_2021.set_index('Time', inplace = True)
"df_diffed_2021 = df_diffed_2021.dropna()"

xls = pd.ExcelFile('Differenced Data 2022.xlsx')
df_diffed_2022 = pd.read_excel(xls)
df_diffed_2022['Time'] = pd.to_datetime(df_diffed_2022['Time'], format='%d/%m/%Y %H:%M:%S')
df_diffed_2022.set_index('Time', inplace = True)
"df_diffed_2022 = df_diffed_2022.dropna()"


df_final = pd.concat([df_diffed_2021, df_diffed_2022])


# Plotting the Data before removing outliers
plt.figure(figsize=(15, 5))
plt.plot(df_final,linewidth=1.5)
plt.title('Raw data before outlier removal',fontweight='bold')
plt.ylabel('Production', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xlabel('Time', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.show()

"df_final = df_final.dropna()"
df_final = df_final.reset_index()

from sklearn.neighbors import LocalOutlierFactor
# fit the model for outlier detection (default)
model = LocalOutlierFactor(n_neighbors=20, contamination='auto')
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = model.fit_predict(df_final.iloc[:,1:])
y_pred_df =  pd.DataFrame(y_pred)
# Detecting the indexes that contain outliers
index = []
for i in range(len(y_pred)):
  if y_pred[i]!=1:
    index.append(i)
index = pd.DataFrame(index)    
# Removing the outliers from the dataframe    
df_final.drop(df_final.index[index[0]], inplace=True)


# Saving the clean dataframe 
file_name = '***REVISED_LOF_Clean_final_twoyears_data_Rescaled.xlsx'
df_final.to_excel(file_name)

# Opening from Excel
xls = pd.ExcelFile('***REVISED_LOF_Clean_final_twoyears_data_Rescaled.xlsx')
df_final = pd.read_excel(xls)
del df_final['Unnamed: 0']
df_final.set_index('Time', inplace = True)
# Data Visualization
df_final[df_final < 0] = 0
"""df_final_2 = df_final[np.abs(df_final.iloc[:,0:8]-df_final.iloc[:,0:8].mean()) <= (3*df_final.iloc[:,0:8].std())]
df_final_2 = df_final.dropna()"""
df_final.plot(linewidth = 0.5).legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)



# Plotting the Data after removing outliers
plt.figure(figsize=(15, 5))

# Plot each column separately to add labels
for column in df_final.columns:
    plt.plot(df_final.index, df_final[column], label=column, linewidth=1.5)

plt.title('Scaled data after outlier removal', fontweight='bold')
plt.ylabel('Energy Production', loc='center', fontsize=20, fontweight='bold')
plt.xlabel('Time', loc='center', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)

# Add legend with labels
plt.legend(loc='best', bbox_to_anchor=(1.0, -0.35), ncol=4)
plt.show()


# Sub plots for 8 inverters
# Plot each column in a separate subplot
axes = df_final.plot(subplots=True, figsize=(12, 12), layout=(4, 2), sharex=True,linewidth=0.3)

# Set common title
plt.suptitle('Production Values for Each Inverter', fontsize=20, fontweight='bold')

# Set axis labels for each subplot
for ax, column in zip(axes.flatten(), df_final.columns):
    ax.set_title(f'{column} Production')
    ax.set_ylabel('Production', fontsize=12, fontweight='bold')
    ax.grid(True)

# Set common x-label only on the bottom subplots
for ax in axes[-1, :]:
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make room for the suptitle
plt.show()





df_inv4 = df_final[['INV/4/DayEnergy (kWh)']]

df_inv4.plot(style='-',
        figsize=(15, 5),
        color=color_pal[0],
        title='INV/4/DayEnergy (kWh)').legend(loc='upper center',bbox_to_anchor=(0.5, -0.35))
plt.show()


























