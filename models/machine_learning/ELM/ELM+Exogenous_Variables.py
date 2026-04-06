#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:53:17 2024

@author: taraneh
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from scipy.linalg import pinv

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


# Split the data into training and testing sets
train_size = int(len(merged_df) * 0.80)  # 80% for training
val_size = int(len(merged_df) * 0.05)   # 5% for validation
test_size = len(merged_df) - train_size - val_size  # Remaining 15% for testing

X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]


# Building the ELM model
input_size = X_train.shape[1]
hidden_size = 100

mu, sigma = 0, 1
w_lo = -1
w_hi = 1
b_lo = -1
b_hi = 1

input_weights = np.random.uniform(w_lo, w_hi, size=(input_size, hidden_size))
biases = np.random.uniform(b_lo, b_hi, size=(hidden_size,))

def relu(x):
    return np.maximum(x, 0, x)

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

output_weights = np.dot(pinv(hidden_nodes(X_train)), y_train)

# Predict
def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(X_test)
prediction_val = predict(X_val)

# Correcting the indexses
index = merged_df.iloc[train_size:]
val_index = index[:val_size]
y_val = pd.DataFrame(y_val)
y_val.index = val_index.index

test_index = index[val_size:]
y_test = pd.DataFrame(y_test)
y_test.index = test_index.index


prediction = pd.DataFrame(prediction)
prediction_val = pd.DataFrame(prediction_val)

prediction.index = test_index.index
prediction_val.index = val_index.index


# Replace negative values with zero
prediction = prediction.applymap(lambda x: 0 if x < 0 else x)


# Plot the forecast
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, linewidth=1.0, label='Actual')
ax.plot(prediction, linewidth=1.0, color='red', label='Predictions')
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.title('Extreme Learning Machine Prediction', fontweight='bold')
plt.xticks(rotation=45)
plt.show()

# Zooming Plot
fig, ax = plt.subplots(figsize = (15,5))
ax.plot(y_test[1500:2500],linewidth=1.0, label = 'Actual')
ax.plot(prediction[1500:2500], linewidth=1.0, color='red',label = 'Predictions')
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production',fontsize=20 ,fontweight='bold')
plt.title('Extreme Learning Machine Prediction (Zoomed)', fontweight='bold')
plt.xticks(rotation=45)

# Evaluate the forecasts
rmse = math.sqrt(mean_squared_error(y_test, prediction))
print('Test RMSE: %.3f' % rmse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test, prediction)

  ### MAPE Score ###
prediction.rename(columns={0: 'INV/4/DayEnergy (kWh)'}, inplace=True)
Error = np.sum(np.abs(np.subtract(y_test,prediction)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, prediction)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, prediction)
print('MSE: %f' % mse)


# Calculate predictions on training and validation data
train_predictions = np.dot(hidden_nodes(X_train), output_weights)
validation_predictions = np.dot(hidden_nodes(X_val), output_weights)

# Calculate mean squared error
train_loss = np.mean((train_predictions - y_train) ** 2)
validation_loss = np.mean((validation_predictions - y_val) ** 2)

print("Training Loss:", train_loss)
print("Validation Loss:", validation_loss)



#------------------------------------------------------------------------------
# Implementation of the Bayesian Optimization

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from scipy.linalg import pinv
import math


# Activation functions
def relu(x):
    return np.maximum(x, 0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Hidden node computation
def hidden_nodes(X, input_weights, biases, activation_func):
    G = np.dot(X, input_weights)
    G = G + biases
    H = activation_func(G)
    return H

# Train ELM with regularization
def train_elm(X_train, y_train, input_weights, biases, activation_func, regularization):
    H = hidden_nodes(X_train, input_weights, biases, activation_func)
    if regularization > 0:
        output_weights = np.dot(pinv(H.T @ H + regularization * np.eye(H.shape[1])) @ H.T, y_train)
    else:
        output_weights = np.dot(pinv(H), y_train)
    return output_weights

def predict_elm(X, input_weights, biases, output_weights, activation_func):
    H = hidden_nodes(X, input_weights, biases, activation_func)
    predictions = np.dot(H, output_weights)
    return predictions

# Objective function for Hyperopt
def objective(params):
    window_size = int(params['window_size'])  # Window size (lag features)
    hidden_size = int(params['hidden_size'])  # Number of hidden nodes
    w_lo = params['w_lo']  # Lower bound for input weights
    w_hi = params['w_hi']  # Upper bound for input weights
    b_lo = params['b_lo']  # Lower bound for biases
    b_hi = params['b_hi']  # Upper bound for biases
    regularization = params['regularization']  # Regularization parameter
    activation_func = params['activation']  # Activation function

    # Create lag features dynamically
    df_inv4 = merged_df[['INV/4/DayEnergy (kWh)']].copy()
    for lag in range(1, window_size + 1):
        df_inv4[f'lag{lag}'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(lag)
    df_inv4 = df_inv4.dropna()

    # Merge lag features with exogenous variables
    lag_features = [f'lag{lag}' for lag in range(1, window_size + 1)]
    FEATURES = lag_features + ['Ambient_temp(degC)', 'Cell_temp(degC)']
    df_final = pd.merge(df_inv4, merged_df[['Ambient_temp(degC)', 'Cell_temp(degC)']], left_index=True, right_index=True)

    X = df_final[FEATURES]
    y = df_final[['INV/4/DayEnergy (kWh)']]

    # Train-test split
    train_size = int(len(df_final) * 0.80)
    val_size = int(len(df_final) * 0.05)
    test_size = len(df_final) - train_size - val_size

    X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]

    # Initialize weights and biases
    input_weights = np.random.uniform(w_lo, w_hi, size=(X_train.shape[1], hidden_size))
    biases = np.random.uniform(b_lo, b_hi, size=(hidden_size,))

    # Train the ELM model
    output_weights = train_elm(X_train, y_train, input_weights, biases, activation_func, regularization)

    # Predict on the validation set
    val_predictions = predict_elm(X_val, input_weights, biases, output_weights, activation_func)

    # Calculate RMSE for validation set
    rmse = math.sqrt(mean_squared_error(y_val, val_predictions))

    return {'loss': rmse, 'status': STATUS_OK}



# Define the search space
space = {
    'window_size': hp.quniform('window_size', 3, 10, 1),  # Lag features: 3 to 10
    'hidden_size': hp.quniform('hidden_size', 50, 200, 1),  # Hidden nodes: 50 to 200
    'w_lo': hp.uniform('w_lo', -1, 0),  # Lower bound for input weights
    'w_hi': hp.uniform('w_hi', 0, 1),  # Upper bound for input weights
    'b_lo': hp.uniform('b_lo', -1, 0),  # Lower bound for biases
    'b_hi': hp.uniform('b_hi', 0, 1),  # Upper bound for biases
    'regularization': hp.loguniform('regularization', -8, 0),  # Regularization: e^-8 to e^0
    'activation': hp.choice('activation', [relu, sigmoid, tanh]),  # Activation functions
}



# Corrected fmin setup
trials = Trials()
best_params = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=50,  # Number of evaluations
                   trials=trials)


print("Best Hyperparameters:", best_params)

# Interpret the best parameters
best_window_size = int(best_params['window_size'])  # Best window size
best_hidden_size = int(best_params['hidden_size'])  # Best hidden size
best_w_lo = best_params['w_lo']  # Best lower bound for input weights
best_w_hi = best_params['w_hi']  # Best upper bound for input weights
best_b_lo = best_params['b_lo']  # Best lower bound for biases
best_b_hi = best_params['b_hi']  # Best upper bound for biases
best_regularization = best_params['regularization']  # Best regularization parameter
best_activation_func = [relu, sigmoid, tanh][best_params['activation']]  # Best activation function

# Plot the Validation
# Extract validation losses from trials
val_losses = [trial['loss'] for trial in trials.results if 'loss' in trial]

# Plot validation losses
plt.figure(figsize=(10, 6))
plt.plot(val_losses, marker='o', linestyle='-')
plt.title('Validation Losses during Hyperparameter Optimization')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss (MSE)')
plt.grid(True)
plt.show()



# =============================================================================
# Using the Optimizeed Hyperparameters: =======================================

best_window_size = 9
best_hidden_size = 200
best_w_lo = -0.7003752852436951
best_w_hi =  0.9927834081536848
best_b_lo = -0.6806278615431923
best_b_hi = 0.997909110229592
best_regularization = 0.00033980100873285546  # Best regularization parameter
best_activation_func = [relu, sigmoid, tanh][0]

# =============================================================================



# Train and evaluate the final model with best parameters
df_inv4 = merged_df[['INV/4/DayEnergy (kWh)']].copy()
for lag in range(1, best_window_size + 1):
    df_inv4[f'lag{lag}'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(lag)
df_inv4 = df_inv4.dropna()

lag_features = [f'lag{lag}' for lag in range(1, best_window_size + 1)]
FEATURES = lag_features + ['Ambient_temp(degC)', 'Cell_temp(degC)']
df_final = pd.merge(df_inv4, merged_df[['Ambient_temp(degC)', 'Cell_temp(degC)']], left_index=True, right_index=True)

X = df_final[FEATURES]
y = df_final[['INV/4/DayEnergy (kWh)']]

train_size = int(len(df_final) * 0.80)
val_size = int(len(df_final) * 0.05)
test_size = len(df_final) - train_size - val_size

X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]

best_input_weights = np.random.uniform(best_w_lo, best_w_hi, size=(X_train.shape[1], best_hidden_size))
best_biases = np.random.uniform(best_b_lo, best_b_hi, size=(best_hidden_size,))

final_output_weights = train_elm(X_train, y_train, best_input_weights, best_biases, best_activation_func, best_regularization)

# Test predictions
final_predictions = predict_elm(X_test, best_input_weights, best_biases, final_output_weights, best_activation_func)

# Replace negative values with zero
final_predictions = final_predictions.applymap(lambda x: 0 if x < 0 else x)

# Evaluate the forecasts
rmse = math.sqrt(mean_squared_error(y_test, final_predictions))
print('Test RMSE: %.3f' % rmse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test, final_predictions)

  ### MAPE Score ###
final_predictions = pd.DataFrame(final_predictions)
final_predictions.index = y_test.index
final_predictions.rename(columns={0: 'INV/4/DayEnergy (kWh)'}, inplace=True)
Error = np.sum(np.abs(np.subtract(y_test,final_predictions)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, final_predictions)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, final_predictions)
print('MSE: %f' % mse)


# Plot the forecast
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, linewidth=1.0, label='Actual')
ax.plot(final_predictions, linewidth=1.0, color='red', label='Predictions')
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.title('Extreme Learning Machine Prediction', fontweight='bold')
plt.xticks(rotation=45)
plt.show()
