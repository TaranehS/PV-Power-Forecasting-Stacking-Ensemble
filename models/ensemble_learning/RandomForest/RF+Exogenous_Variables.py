#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:39:04 2023

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

   ### Creating Lag Features ###
df_inv4['lag1'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(1)
df_inv4['lag2'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(2)
df_inv4['lag3'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(3)
df_inv4['lag4'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(4)
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

# Representing the data with exogenous variables
merged_df[['WTH/5/TAmb (degC)', 'WTH/5/TCell (degC)']].plot(linewidth = 0.4,figsize=(15, 5), title='Cell Temperature and Ambient Temperature').legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)
plt.show()

# Rename the columns 
merged_df.rename(columns={'WTH/5/TAmb (degC)': 'Ambient_temp(degC)'}, inplace=True)
merged_df.rename(columns={'WTH/5/TCell (degC)': 'Cell_temp(degC)'}, inplace=True)


# Modify the feature creation function to incorporate exogenous variables
def create_features_with_exog(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Create features with exogenous variables
merged_df = create_features_with_exog(merged_df)

# Update the features and target
FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3', 'lag4', 'Ambient_temp(degC)', 'Cell_temp(degC)']
TARGET = 'INV/4/DayEnergy (kWh)'

X = merged_df[FEATURES]
y = merged_df[TARGET]

# Split the data into training and testing sets
X_train = X.iloc[:56375]
X_test = X.iloc[56375:]
y_train = y.iloc[:56375]
y_test = y.iloc[56375:]

## Beginning the modeling with Random Forest
from sklearn.ensemble import RandomForestRegressor

# Adjust the model to consider the exogenous variables
model = RandomForestRegressor(n_estimators=100, max_features=5, random_state=1)  # considering the two exogenous variables

# Train the model
model.fit(X_train, y_train)

# Test the model
pred = model.predict(X_test)

# Convert to dataframe to rebuild datetime index
pred = pd.DataFrame(pred)
pred.index = X_test.index
y_test = pd.DataFrame(y_test)
y_test.index = X_test.index

# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Random Forest prediction with time features, lag variables, and exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('RF prediction with exogenous variables (Zoomed)',fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test,pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(y_test,pred)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test, pred)



# =============================================================================
# Setting validation data to get the validation score and plot the model's performance


X = merged_df[FEATURES]
y = merged_df[TARGET]

# Split the data into training and testing sets
train_size = int(len(merged_df) * 0.80)  # 80% for training
val_size = int(len(merged_df) * 0.05)   # 5% for validation
test_size = len(merged_df) - train_size - val_size  # Remaining 15% for testing

X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]

## Beginning the modeling with Random Forest
from sklearn.ensemble import RandomForestRegressor

# Adjust the model to consider the exogenous variables
model = RandomForestRegressor(n_estimators=100, max_features=5, random_state=1)  # considering the two exogenous variables

# Initialize an empty list to store the validation set scores
validation_scores = []

# Training
# Try different numbers of trees
for n_trees in range(1, 101):
    model = RandomForestRegressor(n_estimators=n_trees, max_features=5, random_state=1)
    model.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)
    
    # Compute validation set score (e.g., Mean Squared Error)
    validation_score = mean_squared_error(y_val, y_val_pred)
    validation_scores.append(validation_score)


# Plot the validation set scores
plt.plot(np.arange(1, 101), validation_scores, marker='o', linestyle='-')
plt.xlabel('Number of Trees')
plt.ylabel('Validation Set Score (MSE)')
plt.title('Validation Set Score vs. Number of Trees')
plt.grid(True)
plt.show()


# ((The accuracy of the model prediction reduced in this case comparing to the original case))
# Test the model
pred = model.predict(X_test)
# =============================================================================




# -----------------------------------------------------------------------------
# Implementation of the Bayesian Optimization method on Random Forest for tuning the hyperparameters


from hyperopt import hp, tpe, fmin, Trials, space_eval
from sklearn.model_selection import cross_val_score

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', range(50, 500)),
    'max_features': hp.choice('max_features', range(1, len(FEATURES))),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 20)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 20)),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'random_state': hp.choice('random_state', [1])  # Assuming fixed random state
}

# Define the objective function to minimize (negative mean cross-validation score)
def objective(params):
    model = RandomForestRegressor(**params)
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    # Return the negative mean squared error
    return -cv_scores.mean()

# Initialize Trials object to keep track of results
trials = Trials()

# Perform optimization using TPE algorithm
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

# Get the best hyperparameters
best_params = space_eval(space, best)
best_params = {key: val for key, val in best.items() if key in space}

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Re-train the model with the best hyperparameters
best_model = RandomForestRegressor(**best_params)

best_model = RandomForestRegressor(n_estimators=298, max_features=10, max_depth=14, min_samples_split=13, min_samples_leaf=4, random_state=1, bootstrap = True)


best_model.fit(X_train, y_train)

# Test the model
pred = best_model.predict(X_test)

# Convert to dataframe to rebuild datetime index
pred = pd.DataFrame(pred)
pred.index = X_test.index
y_test = pd.DataFrame(y_test)
y_test.index = X_test.index

# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Random Forest prediction with time features, lag variables, and exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('RF prediction with exogenous variables (Zoomed)',fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test,pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
pred.rename(columns={0: 'INV/4/DayEnergy (kWh)'}, inplace=True)
Error = np.sum(np.abs(np.subtract(y_test,pred)))
Average = np.sum(y_test)
MAPE = Error/Average
print (MAPE)

### MAE Score ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, pred)
print('MAE: %f' % mae)

### MSE Score ###
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred)
print('MSE: %f' % mse)

### R2 Score ###
from sklearn.metrics import r2_score
r2_score(y_test, pred)


# Plot the validation losses during hyperparameter optimization

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








# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# RF + Exogenous Variables + Normalization with : MinMaxScaler class of Sklearn.preprocessing module




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:41:06 2023

@author: taraneh
"""
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

   ### Creating Lag Features ###
df_inv4['lag1'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(1)
df_inv4['lag2'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(2)
df_inv4['lag3'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(3)
df_inv4['lag4'] = df_inv4['INV/4/DayEnergy (kWh)'].shift(4)
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

# Representing the data with exogenous variables
merged_df[['WTH/5/TAmb (degC)', 'WTH/5/TCell (degC)']].plot(linewidth = 0.4,figsize=(15, 5), title='Cell Temperature and Ambient Temperature').legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=2)
plt.show()

# Rename the columns 
merged_df.rename(columns={'WTH/5/TAmb (degC)': 'Ambient_temp(degC)'}, inplace=True)
merged_df.rename(columns={'WTH/5/TCell (degC)': 'Cell_temp(degC)'}, inplace=True)


# Modify the feature creation function to incorporate exogenous variables
def create_features_with_exog(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Create features with exogenous variables
merged_df = create_features_with_exog(merged_df)

# Update the features and target
FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3', 'lag4', 'Ambient_temp(degC)', 'Cell_temp(degC)']
TARGET = ['INV/4/DayEnergy (kWh)']

X = merged_df[FEATURES]
y = merged_df[TARGET]

# Normalization is optional but recommended for Machine Learning approaches as certain 
# activation functions are sensitive to magnitude of numbers. 
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Split the data into training and testing sets

train_size = int(len(merged_df) * 0.80)
test_size = len(merged_df) - train_size
X_train, X_test = X[0:train_size,:], X[train_size :len(merged_df) ,:]
y_train , y_test =  y[0:train_size,:] , y[train_size:len(merged_df),:]


## Beginning the modeling with Random Forest
from sklearn.ensemble import RandomForestRegressor

# Adjust the model to consider the exogenous variables
model = RandomForestRegressor(n_estimators=100, max_features=5, random_state=1)  # considering the two exogenous variables

# Train the model
model.fit(X_train, y_train)

# Test the model
pred = model.predict(X_test)

# Reshape Pred
pred = pred.reshape(-1, 1)

#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

pred = scaler.inverse_transform(pred)
y_test_inverse = scaler.inverse_transform(y_test)


# Correcting the indexses
train_index = merged_df.iloc[:56375]
test_index = merged_df.iloc[56375:]

pred = pd.DataFrame(pred)
pred.index = test_index.index
y_test_inverse = pd.DataFrame(y_test_inverse)
y_test_inverse.index = test_index.index 

'''# Replace negative values with zero
pred = pred.applymap(lambda x: 0 if x < 0 else x)'''

    
# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test_inverse, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('Random Forest prediction with time features, lag variables, and exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test_inverse[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('RF prediction with exogenous variables (Zoomed)',fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Metrics
# RMSE
import math
testScore = math.sqrt(mean_squared_error(y_test_inverse,pred))
print('Test Score: %.2f RMSE' % (testScore))

  ### MAPE Score ###
Error = np.sum(np.abs(np.subtract(y_test_inverse,pred)))
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






























