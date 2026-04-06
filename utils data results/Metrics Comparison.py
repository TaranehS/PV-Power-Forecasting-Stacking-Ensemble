#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:24:38 2024

@author: taraneh
"""

import numpy as np
import matplotlib.pyplot as plt

# Data for the models
models = ['LSTM', 'GRU', 'FFNN', 'ELM', 'RF', 'XGBoost', 'ESN']
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R²']
values = [
    [3.42, 44.91, 6.70, 0.12, 0.979],  # LSTM
    [3.31, 46.76, 6.84, 0.11, 0.978],  # GRU
    [3.86, 58.39, 7.64, 0.12, 0.973],  # FFNN
    [4.33, 76.68, 8.76, 0.14, 0.96], # ELM
    [3.89, 64.96, 8.06, 0.12, 0.971],  # RF
    [3.88, 59.40, 7.71, 0.13, 0.972],  # XGBoost
    [4.99, 77.26, 8.79, 0.13, 0.972]   # ESN
]

# Convert data to a NumPy array for easier manipulation
values = np.array(values)

# Define colors for the bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Plotting
fig, axs = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=False)

# Plot each metric as a separate subplot
for i, metric in enumerate(metrics):
    axs[i].bar(models, values[:, i], color=colors)
    axs[i].set_title(metric, fontsize=16)
    axs[i].tick_params(axis='x', rotation=45, labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)

# Add some text for labels and title
fig.suptitle('Comparison of Evaluation Metrics for Different Models', fontsize=20)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
