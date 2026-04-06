#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:53:07 2024

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

df_inv4 = df_final[['INV/4/DayEnergy (kWh)']]


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
df = df[df.index >= '2021-06-06 05:10:00']

# Merge the dataframes using a full outer join
merged_df = pd.merge(df, df_inv4, left_index=True, right_index=True, how='outer')
merged_df = merged_df.dropna()

# Rename the columns 
merged_df.rename(columns={'WTH/5/TAmb (degC)': 'Ambient_temp(degC)'}, inplace=True)
merged_df.rename(columns={'WTH/5/TCell (degC)': 'Cell_temp(degC)'}, inplace=True)


# Representing the data with exogenous variables
merged_df[['Ambient_temp(degC)', 'Cell_temp(degC)']].plot(linewidth = 0.4,figsize=(15, 5), title='Cell Temperature and Ambient Temperature').legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)
plt.show()


# Update the features and target
FEATURES = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'Ambient_temp(degC)', 'Cell_temp(degC)']
TARGET = ['INV/4/DayEnergy (kWh)']

X = merged_df[FEATURES]
y = merged_df[TARGET]

# Normalization is optional but recommended for neural network as certain 
# activation functions are sensitive to magnitude of numbers. 
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Split the data into training and testing sets

train_size = int(len(merged_df) * 0.80)  # 80% for training
val_size = int(len(merged_df) * 0.05)   # 5% for validation
test_size = len(merged_df) - train_size - val_size  # Remaining 15% for testing

X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]

# Now X_train, y_train are for training, X_test, y_test are for testing, and X_val, y_val are for validation


"""#Convert pandas dataframe to numpy array
dataset = df_inv4.values
dataset = dataset.astype('float32')""" #COnvert values to float



   ### Modeling
print('Build deep model...')
# create and fit dense model
model = Sequential()
model.add(Dense(64, input_dim=7, activation='relu')) #12
model.add(Dense(32, activation='relu'))  #8
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
print(model.summary()) 

   ### Fit the model 
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=2, epochs=100)
    
   ### make predictions
pred = model.predict(X_test)
pred_val = model.predict(X_val)


#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

pred = scaler.inverse_transform(pred)
pred_val = scaler.inverse_transform(pred_val)

y_test_inverse = scaler.inverse_transform(y_test)
y_val_inverse = scaler.inverse_transform(y_val)



# Correcting the indexses

index = merged_df.iloc[train_size:]
val_index = index[:val_size]
y_val = pd.DataFrame(y_val)
y_val.index = val_index.index

test_index = index[val_size:]
y_test = pd.DataFrame(y_test)
y_test.index = test_index.index


pred = pd.DataFrame(pred)
pred_val = pd.DataFrame(pred_val)

pred.index = test_index.index
pred_val.index = val_index.index

y_test_inverse = pd.DataFrame(y_test_inverse)
y_val_inverse = pd.DataFrame(y_val_inverse)

y_test_inverse.index = test_index.index 
y_val_inverse.index = val_index.index 

# Replace negative values with zero
pred = pred.applymap(lambda x: 0 if x < 0 else x)
pred_val = pred_val.applymap(lambda x: 0 if x < 0 else x)


   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test_inverse,pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(pd.DataFrame(y_test_inverse),pd.DataFrame(pred))))
Average = np.sum(y_test_inverse)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_inverse, pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_inverse, pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test_inverse, pred)


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Feed Forward Neural Network prediction with exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test_inverse[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('Feed Forward Neural Network prediction with exogenous variables (Zoomed)',fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.show()


# Visualize the validation results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred_val, label='validation', color='red', linewidth=2.0)
plt.plot(y_val_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Feed Forward Neural Network validation with exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()



# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('FFNN Model training performance')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



# =============================================================================
# Implementation of Bayesian Optimization 

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping

# Define search space for hyperparameters
space = {
    'num_hidden_layers': hp.choice('num_hidden_layers', [1, 2, 3]),
    'num_neurons': hp.choice('num_neurons', [16, 32, 64, 128]),
    'window_size': hp.choice('window_size', [4, 5, 6]),
    'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid', 'linear']),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'SGD']),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'epochs': hp.choice('epochs', [50, 100, 150]),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5)
}

def create_model(params):
    # Extract hyperparameters
    num_neurons = params['num_neurons']
    activation = params['activation']
    num_hidden_layers = params['num_hidden_layers']
    dropout_rate = params['dropout_rate']
    optimizer = params['optimizer']
    learning_rate = params['learning_rate']
    
    # Define the model
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=input_dim, activation=activation))
    for _ in range(num_hidden_layers - 1):  
        model.add(Dense(num_neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    # Compile the model
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['acc'])
    
    return model

def objective(params):
    # Create the model
    model = create_model(params)
    
    # Split data into training, validation, and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate the size of the training and validation sets based on the window size
    train_size = int(len(merged_df) * 0.80)  # 80% for training
    val_size = int(len(merged_df) * 0.05)   # 5% for validation
    test_size = len(merged_df) - train_size - val_size  # Remaining 15% for testing


    # Split the training and validation sets
    X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    
    # Evaluate the model on the validation set
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
    
    return {'loss': val_loss, 'status': STATUS_OK}


# Define input dimension based on the concatenated data
input_dim = X_train.shape[1]

# Run the optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=20, trials=trials)

print("Best hyperparameters:")
print(best)

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



# -----------------------------------------------------------------------------
# Re-train the FFNN model



# Convert indices to their corresponding values
best_activation = ['relu', 'tanh', 'sigmoid', 'linear'][best['activation']]
best_batch_size = [32, 64, 128][best['batch_size']]
best_epochs = [50, 100, 150][best['epochs']]
best_learning_rate = best['learning_rate']
best_num_hidden_layers = best['num_hidden_layers'] + 1  # Add 1 because indices are 0-based
best_num_neurons = [16, 32, 64, 128][best['num_neurons']]
best_optimizer = ['adam', 'rmsprop', 'SGD'][best['optimizer']]
best_window_size = [4, 5, 6][best['window_size']]
best_dropout_rate = best['dropout_rate']

#==============================================================================
# Using obtained best parameters directly

best_activation = 'relu'
best_batch_size = 64
best_epochs = 150
best_learning_rate = 0.0014312764888012894
best_num_hidden_layers = 2
best_num_neurons = 32
best_optimizer = 'adam'
best_window_size = 5
best_dropout_rate = 0.02234058618220991


# =============================================================================


input_dim = len(FEATURES)

# Define and compile the model with modified hyperparameters
model = Sequential()
model.add(Dense(best_num_neurons, input_dim=input_dim, activation=best_activation))  # Changed activation function
for _ in range(best_num_hidden_layers - 1):
    model.add(Dense(best_num_neurons, activation=best_activation))  # Changed activation function
    model.add(Dropout(best_dropout_rate))  # Added dropout layer
model.add(Dense(1))

# Use RMSprop optimizer with the provided learning rate
optimizer = Adam(learning_rate=best_learning_rate)


model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

# Train the model with the modified hyperparameters
history = model.fit(X_train, y_train, batch_size=best_batch_size, epochs=best_epochs, 
                    validation_data=(X_val, y_val), verbose=2)

   ### make predictions
pred = model.predict(X_test)
pred_val = model.predict(X_val)


#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

pred = scaler.inverse_transform(pred)
pred_val = scaler.inverse_transform(pred_val)

y_test_inverse = scaler.inverse_transform(y_test)
y_val_inverse = scaler.inverse_transform(y_val)



# Correcting the indexses

index = merged_df.iloc[train_size:]
val_index = index[:val_size]
y_val = pd.DataFrame(y_val)
y_val.index = val_index.index

test_index = index[val_size:]
y_test = pd.DataFrame(y_test)
y_test.index = test_index.index


pred = pd.DataFrame(pred)
pred_val = pd.DataFrame(pred_val)

pred.index = test_index.index
pred_val.index = val_index.index

y_test_inverse = pd.DataFrame(y_test_inverse)
y_val_inverse = pd.DataFrame(y_val_inverse)

y_test_inverse.index = test_index.index 
y_val_inverse.index = val_index.index 

# Replace negative values with zero
pred = pred.applymap(lambda x: 0 if x < 0 else x)
pred_val = pred_val.applymap(lambda x: 0 if x < 0 else x)


   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test_inverse,pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(pd.DataFrame(y_test_inverse),pd.DataFrame(pred))))
Average = np.sum(y_test_inverse)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_inverse, pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_inverse, pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test_inverse, pred)


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Feed Forward Neural Network prediction with exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test_inverse[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('Feed Forward Neural Network prediction with exogenous variables (Zoomed)',fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.show()


# Visualize the validation results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred_val, label='validation', color='red', linewidth=2.0)
plt.plot(y_val_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Feed Forward Neural Network validation with exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()



# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('FFNN Model training performance after optimization')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



