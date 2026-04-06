#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:16:53 2024

@author: taraneh
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import optuna
import math

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# Preparing the train, validation and test values for the MetaModel
# Ensure that the 'Time' column is a datetime object
y_test_MetaModel = pd.read_csv('y_test_MetaModel.csv')
y_train_MetaModel = pd.read_csv('y_train_MetaModel.csv')
y_val_MetaModel = pd.read_csv('y_validation_MetaModel.csv')

y_test_MetaModel['Time'] = pd.to_datetime(y_test_MetaModel['Time'])
y_train_MetaModel['Time'] = pd.to_datetime(y_train_MetaModel['Time'])
y_val_MetaModel['Time'] = pd.to_datetime(y_val_MetaModel['Time'])

y_test_MetaModel.set_index('Time', inplace=True)
y_train_MetaModel.set_index('Time', inplace=True)
y_val_MetaModel.set_index('Time', inplace=True)

# Step 1: Predictions are generated from the first layers' models on the meta-model training set (base models test sets)
# Load predictions of the individual models from CSV files
lstm_preds = pd.read_csv('LSTM_Base_Test_predictions.csv')
lstm_preds.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
lstm_preds.set_index('Time', inplace=True)
lstm_preds.rename(columns={'0': 'LSTM'}, inplace=True)
lstm_preds.index = pd.to_datetime(lstm_preds.index)

gru_preds = pd.read_csv('GRU_Base_Test_predictions.csv')
gru_preds.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
gru_preds.set_index('Time', inplace=True)
gru_preds.rename(columns={'0': 'GRU'}, inplace=True)
gru_preds.index = pd.to_datetime(gru_preds.index)

ffnn_preds = pd.read_csv('FFNN_Base_Test_predictions.csv')
ffnn_preds.set_index('Time', inplace=True)
ffnn_preds.rename(columns={'0': 'FFNN'}, inplace=True)
ffnn_preds.index = pd.to_datetime(ffnn_preds.index)

rf_preds = pd.read_csv('RF_Base_Test_predictions.csv')
rf_preds.set_index('Time', inplace=True)
rf_preds.rename(columns={'0': 'RF'}, inplace=True)
rf_preds.index = pd.to_datetime(rf_preds.index)

xgb_preds = pd.read_csv('XGBoost_Base_Test_predictions.csv')
xgb_preds.set_index('Time', inplace=True)
xgb_preds.rename(columns={'0': 'XGBoost'}, inplace=True)
xgb_preds.index = pd.to_datetime(xgb_preds.index)

# Making the individual model's predictions for the exact same start and end index
lstm_preds = lstm_preds.iloc[1:]
gru_preds = gru_preds.iloc[1:]
ffnn_preds = ffnn_preds.iloc[1:]
rf_preds = rf_preds.iloc[1:]
xgb_preds = xgb_preds.iloc[1:]

y_train_MetaModel = y_train_MetaModel.iloc[1:]

# Stack predictions as input features for the meta-model
X_train_MetaModel = np.column_stack((lstm_preds, gru_preds, ffnn_preds, rf_preds, xgb_preds))

# Load validation data of the individual models from CSV files
lstm_val_preds = pd.read_csv('LSTM_MetaModel_Val_predictions.csv')
lstm_val_preds.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
lstm_val_preds.set_index('Time', inplace=True)
lstm_val_preds.rename(columns={'0': 'LSTM'}, inplace=True)
lstm_val_preds.index = pd.to_datetime(lstm_val_preds.index)
lstm_val_preds = lstm_val_preds.applymap(lambda x: 0 if x < 0 else x)

gru_val_preds = pd.read_csv('GRU_MetaModel_Val_predictions.csv')
gru_val_preds.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
gru_val_preds.set_index('Time', inplace=True)
gru_val_preds.rename(columns={'0': 'GRU'}, inplace=True)
gru_val_preds.index = pd.to_datetime(gru_val_preds.index)
gru_val_preds = gru_val_preds.applymap(lambda x: 0 if x < 0 else x)

ffnn_val_preds = pd.read_csv('FFNN_MetaModel_Val_predictions.csv')
ffnn_val_preds.set_index('Time', inplace=True)
ffnn_val_preds.rename(columns={'0': 'FFNN'}, inplace=True)
ffnn_val_preds.index = pd.to_datetime(ffnn_val_preds.index)
ffnn_val_preds = ffnn_val_preds.applymap(lambda x: 0 if x < 0 else x)

rf_val_preds = pd.read_csv('RF_MetaModel_Val_predictions.csv')
rf_val_preds.set_index('Time', inplace=True)
rf_val_preds.rename(columns={'0': 'RF'}, inplace=True)
rf_val_preds.index = pd.to_datetime(rf_val_preds.index)
rf_val_preds = rf_val_preds.applymap(lambda x: 0 if x < 0 else x)

xgb_val_preds = pd.read_csv('XGBoost_MetaModel_Val_predictions.csv')
xgb_val_preds.set_index('Time', inplace=True)
xgb_val_preds.rename(columns={'0': 'XGBoost'}, inplace=True)
xgb_val_preds.index = pd.to_datetime(xgb_val_preds.index)
xgb_val_preds = xgb_val_preds.applymap(lambda x: 0 if x < 0 else x)

# Making the individual model's validation predictions for the exact same start and end index
xgb_val_preds = xgb_val_preds.iloc[1:]
rf_val_preds = rf_val_preds.iloc[1:]
ffnn_val_preds = ffnn_val_preds.iloc[1:]
gru_val_preds = gru_val_preds.iloc[1:]
lstm_val_preds = lstm_val_preds.iloc[1:]

y_val_MetaModel = y_val_MetaModel.iloc[1:]

# Stack validation predictions as input features for the meta-model
X_val_MetaModel = np.column_stack((lstm_val_preds, gru_val_preds, ffnn_val_preds, rf_val_preds, xgb_val_preds))

# Here we first generate individual (first layer models) predictions on the final test set (X_test)
# Individual models' predictions on the X_test:
y_test_pred_lstm = pd.read_csv('LSTM_MetaModel_Test_predictions.csv')
y_test_pred_lstm.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
y_test_pred_lstm.set_index('Time', inplace=True)
y_test_pred_lstm.rename(columns={'0': 'LSTM'}, inplace=True)
y_test_pred_lstm.index = pd.to_datetime(y_test_pred_lstm.index)
y_test_pred_lstm = y_test_pred_lstm.applymap(lambda x: 0 if x < 0 else x)

y_test_pred_gru = pd.read_csv('GRU_MetaModel_Test_predictions.csv')
y_test_pred_gru.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
y_test_pred_gru.set_index('Time', inplace=True)
y_test_pred_gru.rename(columns={'0': 'GRU'}, inplace=True)
y_test_pred_gru.index = pd.to_datetime(y_test_pred_gru.index)
y_test_pred_gru = y_test_pred_gru.applymap(lambda x: 0 if x < 0 else x)

y_test_pred_xgb = pd.read_csv('XGBoost_MetaModel_Test_predictions.csv')
y_test_pred_xgb.set_index('Time', inplace=True)
y_test_pred_xgb.rename(columns={'0': 'XGBoost'}, inplace=True)
y_test_pred_xgb.index = pd.to_datetime(y_test_pred_xgb.index)
y_test_pred_xgb = y_test_pred_xgb.applymap(lambda x: 0 if x < 0 else x)

y_test_pred_rf = pd.read_csv('RF_MetaModel_Test_predictions.csv')
y_test_pred_rf.set_index('Time', inplace=True)
y_test_pred_rf.rename(columns={'0': 'RF'}, inplace=True)
y_test_pred_rf.index = pd.to_datetime(y_test_pred_rf.index)
y_test_pred_rf = y_test_pred_rf.applymap(lambda x: 0 if x < 0 else x)

y_test_pred_ffnn = pd.read_csv('FFNN_MetaModel_Test_predictions.csv')
y_test_pred_ffnn.set_index('Time', inplace=True)
y_test_pred_ffnn.rename(columns={'0': 'FFNN'}, inplace=True)
y_test_pred_ffnn.index = pd.to_datetime(y_test_pred_ffnn.index)
y_test_pred_ffnn = y_test_pred_ffnn.applymap(lambda x: 0 if x < 0 else x)

# Making the individual model's predictions for the exact same start and end index
y_test_pred_lstm = y_test_pred_lstm.iloc[1:]
y_test_pred_gru = y_test_pred_gru.iloc[1:]
y_test_pred_xgb = y_test_pred_xgb.iloc[1:]
y_test_pred_rf = y_test_pred_rf.iloc[1:]
y_test_pred_ffnn = y_test_pred_ffnn.iloc[1:]

y_test_MetaModel = y_test_MetaModel.iloc[1:]

# Combine final test predictions
X_test_MetaModel = np.column_stack((y_test_pred_lstm, y_test_pred_gru, y_test_pred_xgb, y_test_pred_rf, y_test_pred_ffnn))

### Optimization (TPE)

# Define objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    num_layers = trial.suggest_int('num_layers', 1, 3)
    units = trial.suggest_int('units', 32, 256, step=32)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    epochs = trial.suggest_int('epochs', 20, 100, step=20)

    # Build the neural network
    model = Sequential()
    model.add(Dense(units, activation=activation, input_shape=(X_train_MetaModel.shape[1],)))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    # Select optimizer
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(
        X_train_MetaModel, y_train_MetaModel,
        validation_data=(X_val_MetaModel, y_val_MetaModel),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Get validation RMSE as the objective to minimize
    val_pred = model.predict(X_val_MetaModel).flatten()
    val_rmse = math.sqrt(mean_squared_error(y_val_MetaModel, val_pred))
    return val_rmse

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print best hyperparameters and validation score
print("Best Hyperparameters:", study.best_params)
print("Best Validation RMSE:", study.best_value)



# -----------------------------------------------------------------------------
# Best parameters
best_params = {
 'num_layers': 1,
 'units': 224,
 'activation': 'relu',
 'optimizer': 'rmsprop',
 'learning_rate': 1.8383367453541725e-05,
 'batch_size': 80,
 'dropout_rate': 0.43302269816025857,
 'epochs': 60}




# Retrain with optimal parameters
best_params = study.best_params
meta_model_nn = Sequential()
meta_model_nn.add(Dense(best_params['units'], activation=best_params['activation'], input_shape=(X_train_MetaModel.shape[1],)))
meta_model_nn.add(Dropout(best_params['dropout_rate']))
for _ in range(best_params['num_layers'] - 1):
    meta_model_nn.add(Dense(best_params['units'], activation=best_params['activation']))
    meta_model_nn.add(Dropout(best_params['dropout_rate']))
meta_model_nn.add(Dense(1, activation='linear'))

optimizer = Adam(learning_rate=best_params['learning_rate']) if best_params['optimizer'] == 'adam' else RMSprop(learning_rate=best_params['learning_rate'])
meta_model_nn.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
meta_model_nn.fit(X_train_MetaModel, y_train_MetaModel, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(X_val_MetaModel, y_val_MetaModel), verbose=1)

# Combine final test predictions
stacked_final_pred_nn = meta_model_nn.predict(X_test_MetaModel)

# Step 4: Evaluate the stacking model on the final test set

# RMSE
stacked_rmse = math.sqrt(mean_squared_error(y_test_MetaModel, stacked_final_pred_nn))
print('Stacking Model Test Score: %.2f RMSE' % (stacked_rmse))

# MAE
stacked_mae = mean_absolute_error(y_test_MetaModel, stacked_final_pred_nn)
print('Stacking Model MAE: %f' % stacked_mae)

# MSE
stacked_mse = mean_squared_error(y_test_MetaModel, stacked_final_pred_nn)
print('Stacking Model MSE: %f' % stacked_mse)

# R2 Score
stacked_r2 = r2_score(y_test_MetaModel, stacked_final_pred_nn)
print('Stacking Model R2 Score: %f' % stacked_r2)

# MAPE
Error = np.sum(np.abs(np.subtract(y_test_MetaModel, stacked_final_pred_nn)))
Average = np.sum(y_test_MetaModel)
stacked_mape = Error / Average
print('Stacking Model MAPE: %f' % stacked_mape)

# Step 5: Apply Model Output Statistics (MOS) for better weighting of base models
# Hyperparameter tuning of alpha for Ridge regression using GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_model = Ridge()
grid_search = GridSearchCV(ridge_model, ridge_params, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_MetaModel, y_train_MetaModel)

# Best alpha value
best_alpha = grid_search.best_params_['alpha']
print("Best alpha for Ridge Regression (MOS):", best_alpha)

# ----------------------------------------------------------------------
best_alpha = 100
# ----------------------------------------------------------------------

# Retrain Ridge model with best alpha
mos_model = Ridge(alpha=best_alpha)
mos_model.fit(X_train_MetaModel, y_train_MetaModel)

# Predict on the test set using MOS
stacked_final_pred_mos = mos_model.predict(X_test_MetaModel)

# Evaluate MOS model
stacked_rmse_mos = math.sqrt(mean_squared_error(y_test_MetaModel, stacked_final_pred_mos))
print('MOS Stacking Model Test Score: %.2f RMSE' % (stacked_rmse_mos))

stacked_r2_mos = r2_score(y_test_MetaModel, stacked_final_pred_mos)
print('MOS Stacking Model R2 Score: %f' % stacked_r2_mos)

# MAE
stacked_mae_mos = mean_absolute_error(y_test_MetaModel, stacked_final_pred_mos)
print('Stacking Model MAE: %f' % stacked_mae_mos)

# MSE
stacked_mse_mos = mean_squared_error(y_test_MetaModel, stacked_final_pred_mos)
print('Stacking Model MSE: %f' % stacked_mse_mos)

# MAPE
Error = np.sum(np.abs(np.subtract(y_test_MetaModel, stacked_final_pred_mos)))
Average = np.sum(y_test_MetaModel)
stacked_mape = Error / Average
print('Stacking Model MAPE: %f' % stacked_mape)








