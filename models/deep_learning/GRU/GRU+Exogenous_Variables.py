#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:54:40 2023

@author: taraneh
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
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


 ## Begin the GRU Model

# Modify the df_to_X_y function to include exogenous variables
def df_to_X_y(df, exog_df, window_size=5):
    df_as_np = df.to_numpy()
    exog_as_np = exog_df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = df_as_np[i:i + window_size]
        exog_row = exog_as_np[i + window_size - 1]
        X.append(row)
        label = df_as_np[i + window_size][-1]  # Becuase the target variable is in the last column
        y.append(label)
    return np.array(X), np.array(y)


# Modify the dataset with exogenous variables
WINDOW_SIZE = 5
X1, y1 = df_to_X_y(merged_df[:70468], merged_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][:70468], WINDOW_SIZE)
X1.shape, y1.shape


   ### train/tes/validation split
X_train, y_train = X1[:57000], y1[:57000]
X_val, y_val = X1[57000:61000], y1[57000:61000]
X_test, y_test = X1[61000:], y1[61000:]
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

# Extracting the index based on X1 and y1 dataframes
new_df = merged_df.iloc[5:-5].copy()

   ### Creating Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Create Model with exogenous variables using GRU
gru_input = Input(shape=(5, 3), name='gru_input')  # we have three time series variables
exog_input = Input(shape=(2,), name='exog_input')  # We have two exogenous variables 

# Add a Dense layer for the exogenous input
exog_dense = Dense(5)(exog_input)  # Adjust the number of units based on the data

# GRU layer for the main data
gru_out = GRU(64)(gru_input)  # Adjust the number of units based on the data

# Concatenate GRU output with the exogenous input
combined = concatenate([gru_out, exog_dense], axis=-1)  # Concatenate after adjusting the shapes

# Add dense layers
x = Dense(8, activation='relu')(combined)
output = Dense(1, activation='linear')(x)

# Create the model with GRU
model_gru = Model(inputs=[gru_input, exog_input], outputs=output)

# Compile the GRU model
model_gru.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Display the summary of the GRU model
model_gru.summary()

# Training the GRU model
X_train_exog, y_train_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][:57000], new_df['INV/4/DayEnergy (kWh)'][:57000]
X_val_exog, y_val_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][57000:61000], new_df['INV/4/DayEnergy (kWh)'][57000:61000]
X_test_exog, y_test_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][61000:], new_df['INV/4/DayEnergy (kWh)'][61000:]

cp1 = ModelCheckpoint('model/', save_best_only=True)
history_gru = model_gru.fit([X_train, X_train_exog], y_train, validation_data=([X_val, X_val_exog], y_val), epochs=40, callbacks=[cp1])

# Plotting the GRU model loss
loss_per_epoch_gru = model_gru.history.history['loss']
plt.plot(range(len(loss_per_epoch_gru)), loss_per_epoch_gru)
plt.ylabel('Loss', loc='center', fontsize=20, fontweight='bold')
plt.xlabel('Epoch', loc='center', fontsize=20, fontweight='bold')
plt.title('Loss per epoch (GRU Model)', fontweight='bold')


# Validation for the GRU model
y_val_pred_gru = model_gru.predict([X_val, X_val_exog])
df_val_gru = pd.DataFrame(y_val_pred_gru)

   ### Corecting the index for validation data
# Extracting the correct index based on X1 and y1
new_df = merged_df.iloc[5:-5].copy()

# for the predicted values
index_df = new_df[57000:61000]
df_val_gru.index = index_df.index

# for the true values
y_val = pd.DataFrame(y_val)
index_df = new_df[57000:61000]
y_val.index = index_df.index


# Plotting the validation for the GRU model
plt.figure(figsize=(15, 5))
plt.plot(y_val, label='True', linewidth=1.5)
plt.plot(df_val_gru, label='Predicted', linewidth=1.5)
plt.title('GRU Model Validation: True vs Predicted', fontweight='bold')
plt.ylabel('Production', loc='center', fontsize=20, fontweight='bold')
plt.xlabel('Time', loc='center', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.show()


# Testing the GRU model
y_test_pred_gru = model_gru.predict([X_test, X_test_exog])
df_test_gru = pd.DataFrame(y_test_pred_gru)

# Correcting the index for the predicted values
index_df = new_df[61000:]
df_test_gru.index = index_df.index

# correcting the index for the true values
y_test = pd.DataFrame(y_test)
index_df = new_df[61000:]
y_test.index = index_df.index

# Plotting the test for the GRU model
plt.figure(figsize=(15, 5))
plt.plot(y_test, label='True', linewidth=1.5)
plt.plot(df_test_gru, label='Predicted', linewidth=1.5)
plt.title('GRU Model Test: True vs Predicted', fontweight='bold')
plt.ylabel('Production', loc='center', fontsize=20, fontweight='bold')
plt.xlabel('Time', loc='center', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.show()



   ### Metrics
# RMSE
testScore = math.sqrt(mean_squared_error(y_test, df_test_gru))
print('Test Score: %.2f RMSE' % (testScore))

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test,df_test_gru)

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(y_test,df_test_gru)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, df_test_gru)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, df_test_gru)
print('MSE: %f' % mse)


# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history_gru.history['loss'])
plt.plot(history_gru.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# -----------------------------------------------------------------------------
# Implementation of the Bayesian Optimization for Hyperparameter Tuning


from hyperopt import hp, fmin, tpe, Trials
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate, Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np

# Define search space
space = {
    'window_size': hp.choice('window_size', [4, 5, 6]),
    'gru_units': hp.choice('gru_units', [32, 64, 128]),
    'dense_units': hp.choice('dense_units', [32, 64, 128]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'epochs': hp.choice('epochs', [20, 30, 40, 50]),
    'batch_size': hp.choice('batch_size', [32, 64]),
    'optimizer': hp.choice('optimizer', ['Adam', 'RMSprop', 'SGD']),
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear']),
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)
}


# Define objective function
def objective(params):
    window_size = params['window_size']
    gru_units = params['gru_units']
    dense_units = params['dense_units']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']
    optimizer = params['optimizer']
    activation = params['activation']
    dropout_rate = params['dropout_rate']
    
    # Adjust X_train and X_val based on window_size
    X_train_windowed = X_train[:, -window_size:, :]
    X_val_windowed = X_val[:, -window_size:, :]
    
    # Define model
    gru_input = Input(shape=(None, 3), name='gru_input')
    exog_input = Input(shape=(2,), name='exog_input')
    exog_dense = Dense(5)(exog_input)
    gru_out = GRU(gru_units)(gru_input)
    combined = concatenate([gru_out, exog_dense], axis=-1)
    x = Dense(dense_units, activation=activation)(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='linear')(x)
    model = Model(inputs=[gru_input, exog_input], outputs=output)
    
    # Compile model
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
    # Train model
    cp1 = ModelCheckpoint('model/', save_best_only=True)
    history = model.fit([X_train_windowed, X_train_exog], y_train, validation_data=([X_val_windowed, X_val_exog], y_val), epochs=epochs, batch_size=batch_size, callbacks=[cp1], verbose=0)
    
    # Return validation loss
    val_loss = np.min(history.history['val_loss'])
    return val_loss

# Perform optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=20, trials=trials)

# Print best hyperparameters
print("Best hyperparameters:", best)



# -----------------------------------------------------------------------------

# Re-train the GRU model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, concatenate, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

# Best hyperparameters obtained from optimization
best_params = {
    'activation': 'relu',  # Map index to activation function
    'batch_size': 32 if best['batch_size'] == 0 else 64,  # Map index to batch size
    'dense_units': 32 if best['dense_units'] == 0 else 64 if best['dense_units'] ==1 else 128,  # Map index to dense units
    'dropout_rate': best['dropout_rate'],
    'epochs': 20 if best['epochs'] == 0 else 30 if best['epochs'] == 1 else 40 if best['epochs'] == 2 else 50,  # Map index to number of epochs
    'learning_rate': best['learning_rate'],
    'gru_units': 32 if best['gru_units'] == 0 else 64 if best['gru_units']== 1 else 128,  # Map index to GRU units
    'optimizer': 'Adam' if best['optimizer'] == 0 else 'RMSprop' if best['optimizer'] == 1 else 'SGD',  # Map index to optimizer
    'window_size': 4 if best['window_size'] == 0 else 5 if best['window_size'] == 1 else 6  # Map index to window size
}


# Define the model architecture with best hyperparameters
gru_input = Input(shape=(best_params['window_size'], 3), name='gru_input')
exog_input = Input(shape=(2,), name='exog_input')
exog_dense = Dense(5)(exog_input)
gru_out = GRU(best_params['gru_units'])(gru_input)
combined = concatenate([gru_out, exog_dense], axis=-1)
x = Dense(best_params['dense_units'], activation=best_params['activation'])(combined)
x = Dropout(best_params['dropout_rate'])(x)
output = Dense(1, activation='linear')(x)
model = Model(inputs=[gru_input, exog_input], outputs=output)

# Compile the model
optimizer = Adam(learning_rate=best_params['learning_rate']) if best_params['optimizer'] == 'Adam' else \
            RMSprop(learning_rate=best_params['learning_rate']) if best_params['optimizer'] == 'RMSprop' else \
            SGD(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Display the summary of the model
model.summary()



# =============================================================================
# Best found hyperparameters
best_params = {
    'activation': 'relu',  # Map index to activation function
    'batch_size': 32,
    'dense_units': 64,
    'dropout_rate': 0.24783,
    'epochs': 40,
    'learning_rate': 0.0002,
    'gru_units': 128,
    'optimizer': 'RMSprop',
    'window_size': 5
}
# Define the model architecture with best hyperparameters
gru_input = Input(shape=(best_params['window_size'], 3), name='gru_input')
exog_input = Input(shape=(2,), name='exog_input')
exog_dense = Dense(5)(exog_input)
gru_out = GRU(best_params['gru_units'])(gru_input)
combined = concatenate([gru_out, exog_dense], axis=-1)
x = Dense(best_params['dense_units'], activation=best_params['activation'])(combined)
x = Dropout(best_params['dropout_rate'])(x)
output = Dense(1, activation='linear')(x)
model = Model(inputs=[gru_input, exog_input], outputs=output)

# Compile the model
optimizer = RMSprop(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
# =============================================================================




# Modify the df_to_X_y function to include exogenous variables
def df_to_X_y(df, exog_df, window_size=5):
    df_as_np = df.to_numpy()
    exog_as_np = exog_df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = df_as_np[i:i + window_size]
        exog_row = exog_as_np[i + window_size - 1]
        X.append(row)
        label = df_as_np[i + window_size][-1]  # Becuase the target variable is in the last column
        y.append(label)
    return np.array(X), np.array(y)


# Modify the dataset with exogenous variables
WINDOW_SIZE = 5
X1, y1 = df_to_X_y(merged_df[:70468], merged_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][:70468], WINDOW_SIZE)
X1.shape, y1.shape


   ### train/tes/validation split
X_train, y_train = X1[:57000], y1[:57000]
X_val, y_val = X1[57000:61000], y1[57000:61000]
X_test, y_test = X1[61000:], y1[61000:]
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

# Extracting the index based on X1 and y1 dataframes
new_df = merged_df.iloc[5:-5].copy()


# Training the GRU model
X_train_exog, y_train_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][:57000], new_df['INV/4/DayEnergy (kWh)'][:57000]
X_val_exog, y_val_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][57000:61000], new_df['INV/4/DayEnergy (kWh)'][57000:61000]
X_test_exog, y_test_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][61000:], new_df['INV/4/DayEnergy (kWh)'][61000:]

cp1 = ModelCheckpoint('model/', save_best_only=True)
history_gru = model.fit([X_train, X_train_exog], y_train, validation_data=([X_val, X_val_exog], y_val), epochs=40, callbacks=[cp1])

# Plotting the GRU model loss
loss_per_epoch_gru = model.history.history['loss']
plt.plot(range(len(loss_per_epoch_gru)), loss_per_epoch_gru)
plt.ylabel('Loss', loc='center', fontsize=20, fontweight='bold')
plt.xlabel('Epoch', loc='center', fontsize=20, fontweight='bold')
plt.title('Loss per epoch (GRU Model)', fontweight='bold')



# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history_gru.history['loss'])
plt.plot(history_gru.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



# Validation for the GRU model
y_val_pred_gru = model.predict([X_val, X_val_exog])
df_val_gru = pd.DataFrame(y_val_pred_gru)


# Replace negative values with zero
df_val_gru = df_val_gru.applymap(lambda x: 0 if x < 0 else x)


   ### Corecting the index for validation data
# Extracting the correct index based on X1 and y1
new_df = merged_df.iloc[5:-5].copy()

# for the predicted values
index_df = new_df[57000:61000]
df_val_gru.index = index_df.index

# for the true values
y_val = pd.DataFrame(y_val)
index_df = new_df[57000:61000]
y_val.index = index_df.index


# Plotting the validation for the GRU model
plt.figure(figsize=(15, 5))
plt.plot(y_val, label='True', linewidth=1.5)
plt.plot(df_val_gru, label='Predicted', linewidth=1.5)
plt.title('GRU Model Validation: True vs Predicted', fontweight='bold')
plt.ylabel('Production', loc='center', fontsize=20, fontweight='bold')
plt.xlabel('Time', loc='center', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.show()


# Testing the GRU model
y_test_pred_gru = model.predict([X_test, X_test_exog])
df_test_gru = pd.DataFrame(y_test_pred_gru)


# Correcting the index for the predicted values
index_df = new_df[61000:]
df_test_gru.index = index_df.index

# correcting the index for the true values
y_test = pd.DataFrame(y_test)
index_df = new_df[61000:]
y_test.index = index_df.index

# Plotting the test for the GRU model
plt.figure(figsize=(15, 5))
plt.plot(y_test, label='True', linewidth=1.5)
plt.plot(df_test_gru, label='Predicted', linewidth=1.5)
plt.title('GRU Model Test: True vs Predicted', fontweight='bold')
plt.ylabel('Production', loc='center', fontsize=20, fontweight='bold')
plt.xlabel('Time', loc='center', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.show()



   ### Metrics
# RMSE
testScore = math.sqrt(mean_squared_error(y_test, df_test_gru))
print('Test Score: %.2f RMSE' % (testScore))

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test,df_test_gru)

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(y_test,df_test_gru)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, df_test_gru)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, df_test_gru)
print('MSE: %f' % mse)



# Plot validation losses
# Extract the validation losses from the Trials object and take absolute values
val_losses = [abs(trial['result']['loss']) for trial in trials.trials]

# Plot the validation losses
plt.figure(figsize=(10, 6))
plt.plot(val_losses, marker='o', linestyle='-')
plt.title('Validation Losses During Hyperparameter Optimization')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss (Mean Squared Error)')
plt.grid(True)
plt.show()





























