#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:36:28 2024

@author: taraneh
"""


import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import xgboost as xgb
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

train_size = int(len(merged_df) * 0.80)  # 80% for training
val_size = int(len(merged_df) * 0.05)   # 5% for validation
test_size = len(merged_df) - train_size - val_size  # Remaining 15% for testing

X_train, X_temp, y_train, y_temp = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100)

   ### Feature Importance ###
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance').legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
plt.ylabel('Input Features', loc= 'center', fontsize=20 ,fontweight='bold')
plt.show()

   ### Forecast on Test ###
pred = reg.predict(X_test)
pred_val = reg.predict(X_val)

# Convert to dataframe to rebuild datetime indexh
pred = pd.DataFrame(pred)
pred.index = X_test.index
y_test = pd.DataFrame(y_test)
y_test.index = X_test.index


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('XGBoost prediction with time features, lag variables, and exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('XGBoost prediction with exogenous variables (Zoomed)',fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20 ,fontweight='bold')
plt.xticks(rotation=45)
plt.show()

# Visualize the validation results

# Convert to dataframe to rebuild datetime indexh
pred_val = pd.DataFrame(pred_val)
pred_val.index = X_val.index
y_val = pd.DataFrame(y_val)
y_val.index = X_val.index


plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred_val, label='Validation', color='red', linewidth=2.0)
plt.plot(y_val, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('XGBoost validation with time features, lag variables, and exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()


# Plot validation performance
results = reg.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
ax.legend()
plt.ylabel('RMSE')
plt.xlabel('Number of Boosting Rounds')
plt.title('XGBoost Model Training Performance')
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





# -----------------------------------------------------------------------------
# Implementation of the Bayesian Optimization method on XGBoost for tuning the hyperparameters


from hyperopt import hp, fmin, tpe, Trials, space_eval
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1500)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'gamma': hp.uniform('gamma', 0.0, 1.0),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'min_child_weight': hp.uniform('min_child_weight', 1, 10),
    'early_stopping_rounds': hp.choice('early_stopping_rounds', range(10, 100))
}

# Objective function to minimize (in this case, negative mean squared error)
def objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['early_stopping_rounds'] = int(params['early_stopping_rounds'])
    
    reg = xgb.XGBRegressor(**params, booster='gbtree', objective='reg:linear')
    
    # Define the validation dataset
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    # Train the model
    reg.fit(X_train, y_train,
            eval_set=eval_set,
            verbose=False)
    
    # Get the best iteration based on early stopping
    best_iteration = reg.best_iteration
    
    # Calculate MSE on the test set
    pred = reg.predict(X_test, ntree_limit=best_iteration)
    mse = mean_squared_error(y_test, pred)
    
    return mse

# Initialize Trials object to track the progress
trials = Trials()

# Run the optimization algorithm (TPE)
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

# Get the best parameters
best_params = space_eval(space, best)

# Retrain the model with the best parameters
best_reg = xgb.XGBRegressor(**best_params, booster='gbtree', objective='reg:linear')
best_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)

# Feature Importance
fi = pd.DataFrame(data=best_reg.feature_importances_,
             index=X_train.columns,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance after optimization').legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
plt.ylabel('Input Features', loc= 'center', fontsize=20 ,fontweight='bold')
plt.show()

# Forecast on Test
pred = best_reg.predict(X_test)

# Convert to dataframe to rebuild datetime indexh
pred = pd.DataFrame(pred)
pred.index = X_test.index
y_test = pd.DataFrame(y_test)
y_test.index = X_test.index


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('XGBoost prediction with time features, lag variables, and exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('XGBoost prediction with exogenous variables (Zoomed)',fontweight='bold')
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


# Plot validation performance
results = best_reg.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
ax.legend()
plt.ylabel('RMSE')
plt.xlabel('Number of Boosting Rounds')
plt.title('XGBoost Model Training Performance after optimization')
plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# XGBoost + Exogenous Variables + Normalization with : MinMaxScaler class of Sklearn.preprocessing module


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:23:14 2023

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
import xgboost as xgb
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


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

   ### Feature Importance ###
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance').legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
plt.ylabel('Input Features', loc= 'center', fontsize=20 ,fontweight='bold')
plt.show()

   ### Forecast on Test ###
pred = reg.predict(X_test)

# Reshape Pred
pred = pred.reshape(-1, 1)

#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test)


# Correcting the indexses
train_index = merged_df.iloc[:56375]
test_index = merged_df.iloc[56375:]

pred = pd.DataFrame(pred)
pred.index = test_index.index
y_test = pd.DataFrame(y_test)
y_test.index = test_index.index 


# Visualize the test results
plt.rcParams["figure.figsize"] = (15, 5)
plt.plot(pred, label='Predictions', color='red', linewidth=2.0)
plt.plot(y_test, label='Actual', linewidth=1.0)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.title('XGBoost prediction with time features, lag variables, and exogenous variables', fontweight='bold')
plt.xlabel('Time', fontsize=20, fontweight='bold')
plt.ylabel('Production', fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

   ### Zoom plot
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred[1500:2500],label='Predictions', color='red', linewidth = 2.0)
plt.plot(y_test[1500:2500],label='Actual', linewidth = 1.0)
plt.legend(loc='upper right',bbox_to_anchor=(0.5, -0.35),ncol=2)
plt.title('XGBoost prediction with exogenous variables (Zoomed)',fontweight='bold')
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




