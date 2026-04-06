#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:39:57 2024

@author: taraneh
"""

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


# READING FINAL CLEAN TWO YEARS Rescaled DATA FROM EXCEL

xls = pd.ExcelFile('REVISED_LOF_Clean_final_twoyears_data_Rescaled.xlsx')
df_final = pd.read_excel(xls)
del df_final['Unnamed: 0']
'''from datetime import datetime as dt
df_final['Time'] = df_final['Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
dates = {date : dt.strptime(date, '%Y-%m-%d %H:%M:%S') for date in df_final['Time'].unique()}
df_final['Time'] = df_final['Time'].apply(lambda v: dates[v])'''
df_final.set_index('Time', inplace = True)
df_final[df_final < 0] = 0
# Data Visualization
'''df_final = df_final[np.abs(df_final.iloc[:,0:8]-df_final.iloc[:,0:8].mean()) <= (3*df_final.iloc[:,0:8].std())]
df_final = df_final.dropna()'''
df_final.plot(linewidth = 0.5).legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)

df_inv4 = df_final[['INV/4/DayEnergy (kWh)']]

df_inv4.plot(style='-',
        figsize=(15, 5),
        color=color_pal[0],
        title='INV/4/DayEnergy (kWh)').legend(loc='upper center',bbox_to_anchor=(0.5, -0.35))
plt.show()

   ### Creating Lag Features ###
df_inv4['lag1'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(1)
df_inv4['lag2'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(2)
df_inv4['lag3'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(3)
df_inv4['lag4'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(4)
df_inv4['lag5'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(5)
df_inv4 = df_inv4.dropna()



# Reading the Exogenous varibale: Ambient Temperature and Cell Temperature
file_path = '/Users/taraneh/Documents/PhD Thesis/Thesis Data/T-Dinamik Konya SPP/Energy Data/2021-2022_Tamb-TCell.txt'
df = pd.read_csv(file_path, sep='\t', names=["Time", "WTH/5/TAmb (degC)", "WTH/5/TCell (degC)"], header=0)

# Fixing the Column names
df.drop('WTH/5/TCell (degC)', axis=1, inplace=True)
df.rename(columns={'WTH/5/TAmb (degC)': 'WTH/5/TCell (degC)'}, inplace=True)
df.rename(columns={'Time': 'WTH/5/TAmb (degC)'}, inplace=True)

# Adjusting the Time Index format
df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M:%S')
df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
df.index = pd.to_datetime(df.index)
df = df.rename_axis('Time')

# Deleting the Nan values
df = df.dropna()

# Make the start point of the df index same as the df_inv4 start point index
df = df[df.index >= '2021-06-06 04:30:00']

# Merge the dataframes using a full outer join
merged_df = pd.merge(df, df_inv4, left_index=True, right_index=True, how='outer')
merged_df = merged_df.dropna()

# Representing the data with exogenous variables
merged_df[['WTH/5/TAmb (degC)', 'WTH/5/TCell (degC)']].plot(linewidth = 0.4,figsize=(15, 5), title='Cell Temperature and Ambient Temperature').legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)
plt.show()

# Rename the columns 
merged_df.rename(columns={'WTH/5/TAmb (degC)': 'Ambient_temp(degC)'}, inplace=True)
merged_df.rename(columns={'WTH/5/TCell (degC)': 'Cell_temp(degC)'}, inplace=True)


# Update the features and target
FEATURES = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'Ambient_temp(degC)', 'Cell_temp(degC)']
TARGET = ['INV/4/DayEnergy (kWh)']

X = merged_df[FEATURES]
y = merged_df[TARGET]

# Define split sizes
train_size = int(len(merged_df) * 0.70)  # 70% for training
remaining_size = len(merged_df) - train_size
val_size = int(remaining_size * 0.05)  # 5% of remaining 30% for base model validation
remaining_test_size = remaining_size - val_size  # Remaining 95% to be divided into 3 sets

# Calculate sizes for test_BaseModel, validation_MetaModel, and test_MetaModel
test_base_size = remaining_test_size // 3  # Dividing the remaining 95% into 3 sets
val_meta_size = test_base_size
test_meta_size = remaining_test_size - test_base_size * 2  # The remainder for test_MetaModel

# Split the data
X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
X_val_base, X_temp, y_val_base, y_temp = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]
X_test_base, X_val_meta, X_test_meta = X_temp[:test_base_size], X_temp[test_base_size:test_base_size*2], X_temp[test_base_size*2:]
y_test_base, y_val_meta, y_test_meta = y_temp[:test_base_size], y_temp[test_base_size:test_base_size*2], y_temp[test_base_size*2:]

# Summary of splits
print(f"Train set size: {len(X_train)}")
print(f"Validation set for BaseModel size: {len(X_val_base)}")
print(f"Test set for BaseModel size: {len(X_test_base)}")
print(f"Validation set for MetaModel size: {len(X_val_meta)}")
print(f"Test set for MetaModel size: {len(X_test_meta)}")

#Indecies
train_index = X_train
val_base_index = X_val_base
test_base_index = X_test_base
val_meta_index = X_val_meta
test_meta_index = X_test_meta

# Normalization is optional but recommended for neural network as certain 
# activation functions are sensitive to magnitude of numbers. 
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)

X_val_base = scaler.fit_transform(X_val_base)
y_val_base = scaler.fit_transform(y_val_base)

X_test_base = scaler.fit_transform(X_test_base)
y_test_base = scaler.fit_transform(y_test_base)

X_val_meta = scaler.fit_transform(X_val_meta)
y_val_meta = scaler.fit_transform(y_val_meta)

X_test_meta = scaler.fit_transform(X_test_meta)
y_test_meta = scaler.fit_transform(y_test_meta)



#************* We have to Run the "ESN CLASS" script first here ***************



# =============================================================================

# Best found hyperparameters
best_params ={
 'n_reservoir': 1105,
 'noise': 0.002813009700070239,
 'sparsity': 0.8292014161114312,
 'spectral_radius': 1.35756476777774,
 'input_scaling': 0.8861075270673359
 }

# =============================================================================


# Train and evaluate the final ESN model using the best hyperparameters
best_esn = ESN(
    n_inputs=X_train.shape[1],
    n_outputs=1,
    n_reservoir=best_params['n_reservoir'],
    spectral_radius=best_params['spectral_radius'],
    sparsity=best_params['sparsity'],
    noise=best_params['noise'],
    input_scaling=best_params['input_scaling'],
    random_state=42
)

# Train the model
best_esn.fit(X_train, y_train.ravel())

# Make predictions
pred_test_base = best_esn.predict(X_test_base)

#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.
pred_test_base = scaler.inverse_transform(pred_test_base)
y_test_base_inverse = scaler.inverse_transform(y_test_base)

# Correcting the indexses
y_test_base_inverse = pd.DataFrame(y_test_base_inverse)
y_test_base_inverse.index = test_base_index.index

pred_test_base = pd.DataFrame(pred_test_base)
pred_test_base.index = test_base_index.index


# Replace negative values with zero
pred_test_base = pred_test_base.applymap(lambda x: 0 if x < 0 else x)



   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test_base_inverse,pred_test_base))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(pd.DataFrame(y_test_base_inverse),pd.DataFrame(pred_test_base))))
Average = np.sum(y_test_base_inverse)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_base_inverse, pred_test_base)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_base_inverse, pred_test_base)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test_base_inverse, pred_test_base)


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred_test_base, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test_base_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Echo State Network prediction after hyperparameter tuning', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()



# Base Validation
pred_val_base = best_esn.predict(X_val_base)
pred_val_base = scaler.inverse_transform(pred_val_base)
y_val_base_inverse = scaler.inverse_transform(y_val_base)
# Correcting the indexses
y_val_base_inverse = pd.DataFrame(y_val_base_inverse)
y_val_base_inverse.index = val_base_index.index
pred_val_base = pd.DataFrame(pred_val_base)
pred_val_base.index = val_base_index.index
# Replace negative values with zero
pred_val_base = pred_val_base.applymap(lambda x: 0 if x < 0 else x)


# MetaModel Validation
pred_val_meta = best_esn.predict(X_val_meta)
pred_val_meta = scaler.inverse_transform(pred_val_meta)
y_val_meta_inverse = scaler.inverse_transform(y_val_meta)
# Correcting the indexses
y_val_meta_inverse = pd.DataFrame(y_val_meta_inverse)
y_val_meta_inverse.index = val_meta_index.index
pred_val_meta = pd.DataFrame(pred_val_meta)
pred_val_meta.index = val_meta_index.index
# Replace negative values with zero
pred_val_meta = pred_val_meta.applymap(lambda x: 0 if x < 0 else x)


# MetaModel Test
pred_test_meta = best_esn.predict(X_test_meta)
pred_test_meta = scaler.inverse_transform(pred_test_meta)
y_test_meta_inverse = scaler.inverse_transform(y_test_meta)
# Correcting the indexses
y_test_meta_inverse = pd.DataFrame(y_test_meta_inverse)
y_test_meta_inverse.index = test_meta_index.index
pred_test_meta = pd.DataFrame(pred_test_meta)
pred_test_meta.index = test_meta_index.index
# Replace negative values with zero
pred_test_meta = pred_test_meta.applymap(lambda x: 0 if x < 0 else x)




