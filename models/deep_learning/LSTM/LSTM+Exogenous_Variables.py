#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:39:24 2023

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


 ## Begin the LSTM Model

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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


# Create Model with exogenous variables
lstm_input = Input(shape=(5, 3), name='lstm_input') # we have three time series variables
exog_input = Input(shape=(2,), name='exog_input')   # We have two exogenous variables 

# Add a Dense layer for the exogenous input
exog_dense = Dense(5)(exog_input)  # Adjust the number of units based on the data

# LSTM layer for the main data
lstm_out = LSTM(64)(lstm_input)

# Concatenate LSTM output with the exogenous input
combined = concatenate([lstm_out, exog_dense], axis=-1)  # Concatenate after adjusting the shapes

# Add dense layers
x = Dense(8, activation='relu')(combined)
output = Dense(1, activation='linear')(x)

# Create the model
model = Model(inputs=[lstm_input, exog_input], outputs=output)

# Compile the model
model.compile(optimizer= Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Display the summary of the model
model.summary()

# Training the model
X_train_exog, y_train_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][:57000], new_df['INV/4/DayEnergy (kWh)'][:57000]
X_val_exog, y_val_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][57000:61000], new_df['INV/4/DayEnergy (kWh)'][57000:61000]
X_test_exog, y_test_exog = new_df[['Ambient_temp(degC)', 'Cell_temp(degC)']][61000:], new_df['INV/4/DayEnergy (kWh)'][61000:]

cp1 = ModelCheckpoint('model/', save_best_only=True)
history = model.fit([X_train, X_train_exog], y_train, validation_data=([X_val, X_val_exog], y_val), epochs=40, callbacks=[cp1])

# Plotting the loss
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.ylabel('Loss', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xlabel('Epoch', loc= 'center', fontsize=20 ,fontweight='bold')
plt.title('Loss per epoch',fontweight='bold')


"""   ### load the best model with least loss
from tensorflow.keras.models import load_model
model1 = load_model('model1/')
"""

# Validation
y_val_pred = model.predict([X_val, X_val_exog])
df_val = pd.DataFrame(y_val_pred)

   ### Corecting the index for validation data
# Extracting the correct index based on X1 and y1
new_df = merged_df.iloc[5:-5].copy()

# for the predicted values
index_df = new_df[57000:61000]
df_val.index = index_df.index

# for the true values
y_val = pd.DataFrame(y_val)
index_df = new_df[57000:61000]
y_val.index = index_df.index


# Calculating the metrics for validation
mse, mae = model.evaluate([X_val, X_val_exog], y_val, verbose=0)
print('Validation Mean Squared Error:', mse)
print('Validation Mean Absolute Error:', mae)

# Plotting the validation
plt.figure(figsize=(15, 5))
plt.plot(y_val, label='True',linewidth=1.5)
plt.plot(df_val, label='Predicted',linewidth=1.5)
plt.title('LSTM Model Validation: True vs Predicted',fontweight='bold')
plt.ylabel('Production', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xlabel('Time', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.show()


# Testing the model
y_test_pred = model.predict([X_test, X_test_exog])
df_test = pd.DataFrame(y_test_pred)

# Correcting the index for the predicted values
index_df = new_df[61000:]
df_test.index = index_df.index

# correcting the index for the true values
y_test = pd.DataFrame(y_test)
index_df = new_df[61000:]
y_test.index = index_df.index


# Plotting the test
plt.figure(figsize=(15, 5))
plt.plot(y_test, label='True',linewidth=1.5)
plt.plot(df_test, label='Predicted',linewidth=1.5)
plt.title('LSTM Model Test: True vs Predicted',fontweight='bold')
plt.ylabel('Production', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xlabel('Time', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.show()


   ### Metrics
# RMSE
testScore = math.sqrt(mean_squared_error(y_test, df_test))
print('Test Score: %.2f RMSE' % (testScore))

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test,df_test)

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(y_test,df_test)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, df_test)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, df_test)
print('MSE: %f' % mse)


# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Model training performance')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()







#------------------------------------------------------------------------------
# Applying Bayesian Optimization

from hyperopt import hp, fmin, tpe, Trials
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Dense, Input, concatenate
from keras.models import Model
import numpy as np

# Define the search space for hyperparameters
space = {
    'lstm_units': hp.choice('lstm_units', [32, 64, 128]),
    'dense_units': hp.choice('dense_units', [32, 64, 128]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'epochs': hp.choice('epochs', [40, 50, 100]),
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
    'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid', 'linear']),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5)
}

# Define a function to train and evaluate the model with given hyperparameters
def train_model(params):
    lstm_input = Input(shape=(5, 3), name='lstm_input')
    exog_input = Input(shape=(2,), name='exog_input')

    exog_dense = Dense(params['dense_units'])(exog_input)
    lstm_out = LSTM(params['lstm_units'])(lstm_input)
    combined = concatenate([lstm_out, exog_dense], axis=-1)

    x = Dense(params['dense_units'], activation=params['activation'])(combined)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=[lstm_input, exog_input], outputs=output)

    optimizer = Adam(learning_rate=params['learning_rate']) if params['optimizer'] == 'adam' else RMSprop(learning_rate=params['learning_rate'])

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    cp1 = ModelCheckpoint('model/', save_best_only=True)
    history = model.fit([X_train, X_train_exog], y_train, validation_data=([X_val, X_val_exog], y_val), epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[cp1])

    # Return the validation loss
    val_loss = np.min(history.history['val_loss'])
    return val_loss

# Set the number of optimization iterations
max_evals = 20

# Perform optimization
trials = Trials()
best = fmin(train_model, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

# Print best hyperparameters
print("Best hyperparameters:", best)


# Plot validation losses
# Extract the validation losses from the Trials object and take absolute values
val_losses = [abs(trial['result']['loss']) for trial in trials.trials]
# Plot the validation losses
plt.figure(figsize=(10, 6))
plt.plot(val_losses, marker='o', linestyle='-')
plt.title('Validation Losses During Hyperparameter Optimization for LSTM model')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss (Mean Squared Error)')
plt.grid(True)
plt.show()



# Mapping for each parameter
lstm_units_options = [32, 64, 128]
dense_units_options = [32, 64, 128]
epochs_options = [40, 50, 100]
batch_size_options = [32, 64, 128]
optimizer_options = ['adam', 'rmsprop']
activation_options = ['relu', 'tanh', 'sigmoid', 'linear']


# Retrieve the best parameters and map them to their values
best_params = {
    'lstm_units': lstm_units_options[best['lstm_units']],
    'dense_units': dense_units_options[best['dense_units']],
    'learning_rate': best['learning_rate'],
    'epochs': epochs_options[best['epochs']],
    'batch_size': batch_size_options[best['batch_size']],
    'optimizer': optimizer_options[best['optimizer']],
    'activation': activation_options[best['activation']],
    'dropout_rate': best['dropout_rate']
    }


# =============================================================================

# Best found hyperparameters
best_params = {
    'activation': 'sigmoid',  # Map index to activation function
    'batch_size': 64,
    'dense_units': 128,
    'dropout_rate': 0.100445,
    'epochs': 50,
    'learning_rate': 0.0005,
    'lstm_units': 128,
    'optimizer': 'Adam',
}

# =============================================================================


# Retrain the model with the best hyperparameters
lstm_input = Input(shape=(5, 3), name='lstm_input')
exog_input = Input(shape=(2,), name='exog_input')

exog_dense = Dense(best_params['dense_units'])(exog_input)
lstm_out = LSTM(best_params['lstm_units'])(lstm_input)
combined = concatenate([lstm_out, exog_dense], axis=-1)

x = Dense(best_params['dense_units'], activation=best_params['activation'])(combined)
output = Dense(1, activation='linear')(x)

model = Model(inputs=[lstm_input, exog_input], outputs=output)

optimizer = Adam(learning_rate=best_params['learning_rate']) if best_params['optimizer'] == 'adam' else RMSprop(learning_rate=best_params['learning_rate'])

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

cp1 = ModelCheckpoint('model/', save_best_only=True)
history = model.fit([X_train, X_train_exog], y_train, validation_data=([X_val, X_val_exog], y_val), epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[cp1])

# Return the validation loss
val_loss = np.min(history.history['val_loss'])

# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Model training performance after optimization')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# Testing the model
y_test_pred = model.predict([X_test, X_test_exog])
df_test = pd.DataFrame(y_test_pred)

# Correcting the index for the predicted values
index_df = new_df[61000:]
df_test.index = index_df.index

# correcting the index for the true values
y_test = pd.DataFrame(y_test)
index_df = new_df[61000:]
y_test.index = index_df.index


# Plotting the test
plt.figure(figsize=(15, 5))
plt.plot(y_test, label='True',linewidth=1.5)
plt.plot(df_test, label='Predicted',linewidth=1.5)
plt.title('LSTM Model Test: True vs Predicted',fontweight='bold')
plt.ylabel('Production', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xlabel('Time', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.show()


   ### Metrics
# RMSE
testScore = math.sqrt(mean_squared_error(y_test, df_test))
print('Test Score: %.2f RMSE' % (testScore))

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test,df_test)

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(y_test,df_test)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, df_test)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, df_test)
print('MSE: %f' % mse)










# =============================================================================
# Implementing the LSTM model on the embedded data ============================
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam, RMSprop

# Define the LSTM model
def create_lstm_model(input_shape, lstm_units, dense_units, dropout_rate, activation, learning_rate, optimizer):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=input_shape))
    model.add(Dense(units=dense_units, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))  # Output layer with 1 neuron for regression task

    # Compile model
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")
    
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

# Split the embedded data into features (X) and target (y)
X = embedded_data[:, :-1]
y = embedded_data[:, -1]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 5% of total data for validation, 15% for test


# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (samples, time steps, features)
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = np.reshape(X_val_scaled, (X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Create the LSTM model
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])  # Shape of input for LSTM
lstm_model = create_lstm_model(input_shape=input_shape,
                                lstm_units=best_params['lstm_units'],
                                dense_units=best_params['dense_units'],
                                dropout_rate=best_params['dropout_rate'],
                                activation=best_params['activation'],
                                learning_rate=best_params['learning_rate'],
                                optimizer=best_params['optimizer'])

# Train the model
history = lstm_model.fit(X_train_reshaped, y_train,
                         epochs=best_params['epochs'],
                         batch_size=best_params['batch_size'],
                         validation_data=(X_val_reshaped, y_val))


# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Model training performance on the embedded data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# Predict the model on the test data
y_pred = lstm_model.predict(X_test_reshaped)

# Replace negative values with zero
y_pred = pd.DataFrame(y_pred)
y_pred = y_pred.applymap(lambda x: 0 if x < 0 else x)


# Plotting the test
plt.figure(figsize=(15, 5))
plt.plot(y_test, label='True',linewidth=1.5)
plt.plot(y_pred, label='Predicted',linewidth=1.5)
plt.title('LSTM Model for the embedded data',fontweight='bold')
plt.ylabel('Production', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xlabel('Time', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.show()


   ### Metrics
# RMSE
testScore = math.sqrt(mean_squared_error(y_test, y_pred))
print('Test Score: %.2f RMSE' % (testScore))

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# Calculate MAPE
def calculate_mape(y_true, y_pred):
    mask = y_true != 0  # Create a mask for non-zero true values
    masked_y_true = y_true[mask]
    masked_y_pred = y_pred[mask]
    error = np.sum(np.abs(np.subtract(masked_y_true, masked_y_pred)))
    average = np.sum(masked_y_true)
    return error / average
# Reshape y_pred and y_test to match shape for calculation
y_pred = np.squeeze(y_pred)
y_test = np.squeeze(y_test)
# Calculate MAPE
mape = calculate_mape(y_test, y_pred)
print("Mean Absolute Percentage Error (MAPE):", mape)


### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('MSE: %f' % mse)




# =============================================================================
# Optimize the LSTM model on the embedded data ============================
# =============================================================================


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tensorflow.keras.optimizers import Adam, RMSprop

# Define the search space for hyperparameters
space = {
    'lstm_units': hp.choice('lstm_units', [64, 128, 256]),
    'dense_units': hp.choice('dense_units', [64, 128, 256]),
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
    'activation': hp.choice('activation', ['relu', 'sigmoid']),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'optimizer': hp.choice('optimizer', ['Adam', 'RMSprop']),  # Use optimizer name
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'epochs': hp.choice('epochs', [50, 100, 150])
}

# Define the LSTM model
def create_lstm_model(input_shape, lstm_units, dense_units, dropout_rate, activation, optimizer):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=input_shape))
    model.add(Dense(units=dense_units, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))  # Output layer with 1 neuron for regression task

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Objective function to minimize (loss)
def objective(params):
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    if params['optimizer'] == 'Adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    else:
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    
    model = create_lstm_model(input_shape=input_shape,
                               lstm_units=params['lstm_units'],
                               dense_units=params['dense_units'],
                               dropout_rate=params['dropout_rate'],
                               activation=params['activation'],
                               optimizer=optimizer)  # Pass optimizer directly
    
    history = model.fit(X_train_reshaped, y_train,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        validation_data=(X_val_reshaped, y_val),
                        verbose=0)
    
    val_loss = np.min(history.history['val_loss'])  # Get the minimum validation loss
    
    return {'loss': val_loss, 'status': STATUS_OK}

# Initialize trials object
trials = Trials()

# Run the hyperparameter optimization using TPE
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,  # You can increase this for a more exhaustive search
            trials=trials)

print("Best hyperparameters:", best)


# Plot validation losses
# Extract the validation losses from the Trials object and take absolute values
val_losses = [abs(trial['result']['loss']) for trial in trials.trials]
# Plot the validation losses
plt.figure(figsize=(10, 6))
plt.plot(val_losses, marker='o', linestyle='-')
plt.title('Validation Losses During Hyperparameter Optimization for LSTM model')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss (Mean Squared Error)')
plt.grid(True)
plt.show()


# Re-train the LSTM model
# Split the embedded data into features (X) and target (y)
X = embedded_data[:, :-1]
y = embedded_data[:, -1]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 5% of total data for validation, 15% for test


# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (samples, time steps, features)
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = np.reshape(X_val_scaled, (X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))



# Best found hyperparameters
best_params = {
    'activation': 'relu',  # Map index to activation function
    'batch_size': 128,
    'dense_units': 256,
    'dropout_rate': 0.3136355354741743,
    'epochs': 150,
    'learning_rate': 0.00036987665007314526,
    'lstm_units': 256,
    'optimizer': 'Adam',
}

# Define the LSTM model
def create_lstm_model(input_shape, lstm_units, dense_units, dropout_rate, activation, learning_rate, optimizer):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=input_shape))
    model.add(Dense(units=dense_units, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))  # Output layer with 1 neuron for regression task

    # Compile model
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")
    
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


# Create the LSTM model
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])  # Shape of input for LSTM
lstm_model = create_lstm_model(input_shape=input_shape,
                                lstm_units=best_params['lstm_units'],
                                dense_units=best_params['dense_units'],
                                dropout_rate=best_params['dropout_rate'],
                                activation=best_params['activation'],
                                learning_rate=best_params['learning_rate'],
                                optimizer=best_params['optimizer'])

# Train the model
history = lstm_model.fit(X_train_reshaped, y_train,
                         epochs=best_params['epochs'],
                         batch_size=best_params['batch_size'],
                         validation_data=(X_val_reshaped, y_val))


# Plot training & validation loss
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Model training performance on the embedded data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# Predict the model on the test data
y_pred = lstm_model.predict(X_test_reshaped)

# Replace negative values with zero
y_pred = pd.DataFrame(y_pred)
y_pred = y_pred.applymap(lambda x: 0 if x < 0 else x)


# Plotting the test
plt.figure(figsize=(15, 5))
plt.plot(y_test, label='True',linewidth=1.5)
plt.plot(y_pred, label='Predicted',linewidth=1.5)
plt.title('LSTM Model for the embedded data',fontweight='bold')
plt.ylabel('Production', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xlabel('Time', loc= 'center', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.show()


   ### Metrics
# RMSE
testScore = math.sqrt(mean_squared_error(y_test, y_pred))
print('Test Score: %.2f RMSE' % (testScore))

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# Calculate MAPE
Error = np.sum(np.abs(np.subtract(y_test,y_pred)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('MSE: %f' % mse)


