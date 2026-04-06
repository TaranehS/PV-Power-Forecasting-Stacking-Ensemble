#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:39:41 2024

@author: taraneh
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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





# False Nearest Neighbors Analysis ============================================

from sklearn.neighbors import NearestNeighbors

def false_nearest_neighbors(data, m, tau, threshold):
    # Embedding
    N = len(data)
    embedded_data = np.array([data[i:i+m] for i in range(N-(m-1)*tau)])
    
    # Reshape embedded_data to be a 2D array
    embedded_data = np.reshape(embedded_data, (len(embedded_data), -1))
    
    # Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_data)
    distances, _ = nbrs.kneighbors()
    original_distances = distances[:, 1]
    
    # Perturb each point and calculate distances again
    perturbed_data = embedded_data + np.random.normal(0, 0.1, embedded_data.shape) # Perturb the embedded data
    nbrs_perturbed = NearestNeighbors(n_neighbors=2).fit(perturbed_data)
    distances_perturbed, _ = nbrs_perturbed.kneighbors()
    perturbed_distances = distances_perturbed[:, 1]
    
    # Calculate false nearest neighbors
    false_nn_count = np.sum(perturbed_distances / original_distances > threshold)
    fnn_ratio = false_nn_count / len(data)
    
    return embedded_data, fnn_ratio


# Optimization for embedding dimension

def find_optimal_embedding_dimension(data, max_m, tau, threshold):
    fnn_ratios = []
    for m in range(1, max_m + 1):
        fnn_ratio = false_nearest_neighbors(data, m, tau, threshold)
        fnn_ratios.append(fnn_ratio)
        # Check for stabilization or minimum FNN ratio
        if len(fnn_ratios) > 2 and fnn_ratios[-1] > fnn_ratios[-2]:
            optimal_m = m - 1
            break
    else:
        # If FNN ratio continues to decrease with increasing m, use the maximum m
        optimal_m = max_m
    
    return optimal_m


# Example usage to find optimal embedding dimension
max_m = 100  # Maximum embedding dimension to consider
tau = 1  # time delay
threshold = 1.3  # threshold for identifying false nearest neighbors
optimal_m = find_optimal_embedding_dimension(merged_df.values, max_m, tau, threshold)
print("Optimal embedding dimension:", optimal_m)



# Example usage to embed the time series data 
m = 70  # obtained optimum embedding dimension
tau = 1  # time delay
threshold = 1.3  # threshold for identifying false nearest neighbors

embedded_data, fnn_ratio = false_nearest_neighbors(merged_df.values, m, tau, threshold)
print("False nearest neighbors ratio:", fnn_ratio)
print("Embedded data shape:", embedded_data.shape)


# Visualization ==============================================================

import matplotlib.pyplot as plt

# Define a range of embedding dimensions to explore
max_m = 100
embedding_dimensions = range(1, max_m + 1)

# Calculate FNN ratio for each embedding dimension
fnn_ratios = [false_nearest_neighbors(merged_df.values, m, tau, threshold) for m in embedding_dimensions]

# Plot FNN ratio vs. embedding dimension
plt.figure(figsize=(15, 5))
plt.plot(embedding_dimensions, fnn_ratios, marker='o')
plt.xlabel('Embedding Dimension (m)', fontsize=20 ,fontweight='bold')
plt.ylabel('False Nearest Neighbors Ratio', fontsize=20 ,fontweight='bold')
plt.title('False Nearest Neighbors Ratio vs. Embedding Dimension', fontweight='bold')
plt.grid(True)
plt.show()


# Visualizing Embedded Phase Space 
from sklearn.decomposition import PCA

# Define embedding dimension and time delay
m = 70  # embedding dimension
tau = 1  # time delay

# Embed the time series data
N = len(merged_df)
embedded_data = np.array([merged_df[i:i+m].values for i in range(N-(m-1)*tau)])

# Reshape embedded_data to be a 2D array
embedded_data = np.reshape(embedded_data, (len(embedded_data), -1))

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
embedded_data_pca = pca.fit_transform(embedded_data)

# Plot the embedded phase space
plt.figure(figsize=(15, 5))
plt.scatter(embedded_data_pca[:, 0], embedded_data_pca[:, 1], s=10)
plt.xlabel('Principal Component 1', fontsize=20 ,fontweight='bold')
plt.ylabel('Principal Component 2', fontsize=20 ,fontweight='bold')
plt.title('Embedded Phase Space (PCA)', fontweight='bold')
plt.grid(True)
plt.show()







# Implementation of ESN =======================================================



# Update the features and target
FEATURES = ['Ambient_temp(degC)', 'Cell_temp(degC)']
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


from sklearn.linear_model import Ridge
import numpy as np


#************* We have to Run the "ESN CLASS" script first here ***************



# Instantiate the ESN with desired parameters
esn = ESN(n_inputs=X_train.shape[1], n_outputs=1, n_reservoir=200,
          spectral_radius=0.95, sparsity=0, noise=0.001, input_scaling=None,
          teacher_forcing=True, out_activation=np.tanh, inverse_out_activation=np.tanh,
          random_state=42, silent=False)


def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(15, 5))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_esn(X_train, y_train, X_val, y_val, n_reservoir=1000, sparsity=0.2, spectral_radius=0.9, noise=0.001, n_epochs=20):
    esn = ESN(n_inputs=X_train.shape[1], n_outputs=1, n_reservoir=n_reservoir, sparsity=sparsity, spectral_radius=spectral_radius, noise=noise)
    
    train_loss = []
    val_loss = []
    
    for epoch in range(1, n_epochs + 1):
        # Train the ESN
        esn.fit(X_train, y_train)
        
        # Calculate training loss
        y_train_pred = esn.predict(X_train)
        train_loss.append(mean_squared_error(y_train, y_train_pred))
        
        # Validate the ESN
        y_val_pred = esn.predict(X_val)
        
        # Calculate validation loss
        val_loss.append(mean_squared_error(y_val, y_val_pred))
        
        # Print progress
        print(f'Epoch {epoch}/{n_epochs} - Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}')
    
    # Plot loss
    plot_loss(train_loss, val_loss)
    
    return esn

# Train the ESN
esn_model = train_esn(X_train, y_train, X_val, y_val)

# Predict on the test data
y_test_pred = esn_model.predict(X_test)

#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_inverse = scaler.inverse_transform(y_test)

# Correcting the indexses

index = merged_df.iloc[train_size:]
test_index = index[val_size:]
y_test = pd.DataFrame(y_test)
y_test.index = test_index.index

y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred.index = test_index.index

y_test_inverse = pd.DataFrame(y_test_inverse)
y_test_inverse.index = test_index.index 

# Replace negative values with zero
y_test_pred = y_test_pred.applymap(lambda x: 0 if x < 0 else x)



   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test_inverse,y_test_pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(pd.DataFrame(y_test_inverse),pd.DataFrame(y_test_pred))))
Average = np.sum(y_test_inverse)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_inverse, y_test_pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_inverse, y_test_pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test_inverse, y_test_pred)


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(y_test_pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Echo State Network prediction', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()


# =============================================================================
# Hyperparameter Tuning using TPE =============================================
# =============================================================================


from hyperopt import hp, tpe, Trials, fmin
from sklearn.metrics import mean_squared_error
from hyperopt import space_eval
import numpy as np

# Define the search space for hyperparameters
space = {
    'n_reservoir': hp.choice('n_reservoir', [100, 200, 500, 1000]),
    'sparsity': hp.uniform('sparsity', 0, 0.5),
    'spectral_radius': hp.uniform('spectral_radius', 0.1, 1.0),
    'noise': hp.loguniform('noise', np.log(0.0001), np.log(0.1))
}

def objective(params):
    # Train the ESN with given hyperparameters
    esn = ESN(n_inputs=X_train.shape[1], n_outputs=1, 
              n_reservoir=params['n_reservoir'], 
              sparsity=params['sparsity'], 
              spectral_radius=params['spectral_radius'], 
              noise=params['noise'])
    
    train_loss = []
    val_loss = []
    n_epochs=20
    
    for epoch in range(1, n_epochs + 1):
        # Train the ESN
        esn.fit(X_train, y_train)
        
        # Calculate training loss
        y_train_pred = esn.predict(X_train)
        train_loss.append(mean_squared_error(y_train, y_train_pred))
        
        # Validate the ESN
        y_val_pred = esn.predict(X_val)
        
        # Calculate validation loss
        val_loss.append(mean_squared_error(y_val, y_val_pred))
        
    # Return the validation loss as the objective value
    return np.mean(val_loss)

# Initialize the Trials object to keep track of results
trials = Trials()

# Define the number of optimization rounds
max_evals = 50

# Run the hyperparameter optimization using TPE
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

# Get the best hyperparameters
best_params = space_eval(space, best)
print("Best hyperparameters:", best_params)

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
# =============================================================================

# Best found hyperparameters
best_params ={
 'n_reservoir': 1000,
 'noise': 0.0007306041461134312,
 'sparsity': 0.40332672307701645,
 'spectral_radius': 0.840328301612574
 }

# =============================================================================


# Train the ESN with the best hyperparameters
esn_model = ESN(n_inputs=X_train.shape[1], n_outputs=1, 
                n_reservoir=best_params['n_reservoir'], 
                sparsity=best_params['sparsity'], 
                spectral_radius=best_params['spectral_radius'], 
                noise=best_params['noise'])

# Fit the model with training data
esn_model.fit(X_train, y_train)

# Predict on the test data
y_test_pred = esn_model.predict(X_test)


#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_inverse = scaler.inverse_transform(y_test)

# Correcting the indexses

index = merged_df.iloc[train_size:]
test_index = index[val_size:]
y_test = pd.DataFrame(y_test)
y_test.index = test_index.index

y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred.index = test_index.index

y_test_inverse = pd.DataFrame(y_test_inverse)
y_test_inverse.index = test_index.index 

# Replace negative values with zero
y_test_pred = y_test_pred.applymap(lambda x: 0 if x < 0 else x)



   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test_inverse,y_test_pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(pd.DataFrame(y_test_inverse),pd.DataFrame(y_test_pred))))
Average = np.sum(y_test_inverse)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_inverse, y_test_pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_inverse, y_test_pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test_inverse, y_test_pred)


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(y_test_pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Echo State Network prediction after hyperparameter tuning', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()



# =============================================================================
# Implementation of Echo State Network for the embedded data ==================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyESN import ESN

# Split the embedded data into features (X) and target (y)
X = embedded_data[:, :-1]
y = embedded_data[:, -1]

# Normalize the features
#scaler = MinMaxScaler()
#X_normalized = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 5% of total data for validation, 15% for test

# Initialize and train the ESN model
esn = ESN(n_inputs=X_train.shape[1], n_outputs=1, n_reservoir=2000)
esn.fit(X_train, y_train, inspect=True)

# Validate the trained model
y_val_pred = esn.predict(X_val)

# Calculate validation performance metrics
val_mse = np.mean((y_val_pred - y_val)**2)

print("Validation MSE:", val_mse)

# Perform forecasting with the trained model
y_test_pred = esn.predict(X_test)

# Replace negative values with zero
y_test_pred = pd.DataFrame(y_test_pred).applymap(lambda x: 0 if x < 0 else x)

   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test,y_test_pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(pd.DataFrame(y_test),pd.DataFrame(y_test_pred))))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_test_pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test, y_test_pred)


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(y_test_pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Echo State Network prediction using embedded data', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

# =============================================================================
# Hyperparameter Tuning using TPE =============================================
# =============================================================================

from hyperopt import hp, fmin, tpe, Trials, space_eval
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Define the search space for hyperparameters
space = {
    'n_reservoir': hp.choice('n_reservoir', np.arange(100, 2001, dtype=int)),
    'spectral_radius': hp.uniform('spectral_radius', 0.0, 1.0),
    'sparsity': hp.uniform('sparsity', 0.0, 1.0),
    'noise': hp.uniform('noise', 0.0, 0.5)
}

# Define the objective function to minimize (in our case, validation MSE)
def objective(params):
    esn = ESN(n_inputs=X_train.shape[1], n_outputs=1, **params)
    esn.fit(X_train, y_train)
    y_val_pred = esn.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    return mse

# Split the embedded data into features (X) and target (y)
X = embedded_data[:, :-1]
y = embedded_data[:, -1]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 5% of total data for validation, 15% for test

# Perform the hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Get the best hyperparameters
best_params = space_eval(space, best)
print("Best hyperparameters:", best_params)

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
# =============================================================================

# Best found hyperparameters
best_params ={'n_reservoir': 778,
 'noise': 0.05394331893875537,
 'sparsity': 0.020905396442202795,
 'spectral_radius': 0.35720010281808123}
# =============================================================================

# Train the final model with the best hyperparameters
best_esn = ESN(n_inputs=X_train.shape[1], n_outputs=1, **best_params)
best_esn.fit(X_train, y_train)

# Evaluate the final model on the test set
y_test_pred = best_esn.predict(X_test)

# Replace negative values with zero
y_test_pred = pd.DataFrame(y_test_pred).applymap(lambda x: 0 if x < 0 else x)

   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test,y_test_pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(pd.DataFrame(y_test),pd.DataFrame(y_test_pred))))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_test_pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test, y_test_pred)


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(y_test_pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Echo State Network prediction for embedded data after the hyperparameter tuning', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()
