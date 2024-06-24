#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:57:49 2023

@author: hatai
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Specify the parent directory where subfolders are located
parent_directory = 'result'

# Initialize a dictionary to compile row means for each feature
row_means_compiled = {}

# Iterate through each subfolder and process each feature's `_row_means.xlsx`
for subfolder in os.listdir(parent_directory):
    subfolder_path = os.path.join(parent_directory, subfolder)
    if os.path.isdir(subfolder_path) and subfolder != '.DS_Store':
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('_row_means.xlsx'):
                feature = file_name.replace('_row_means.xlsx', '')
                file_path = os.path.join(subfolder_path, file_name)
                feature_df = pd.read_excel(file_path, engine='openpyxl')
                mean_value = feature_df['Mean'].mean()
                row_means_compiled.setdefault(feature, []).append(mean_value)

# Initialize a list to hold the slope and p-value for each feature
slope_p_values_list = []

# Perform linear regression for each feature based on the compiled row means
for feature, means in row_means_compiled.items():
    if len(means) > 1:  # Ensure there are enough data points for regression
        x_values = np.arange(len(means))
        slope, _, _, p_value, _ = linregress(x_values, means)
        
        # Check if the slope is near zero and the p-value is small enough
        if abs(slope) < 0.01 and p_value < 0.05:  # Adjust thresholds as necessary
            slope_p_values_list.append({'Feature': feature, 'Slope': slope, 'P-Value': p_value})

# Convert the list to a DataFrame
slope_p_values_df = pd.DataFrame(slope_p_values_list)

# Sort the features by their p-values
sorted_features = slope_p_values_df.sort_values('P-Value')

# Output the features with near-zero slopes and statistically significant p-values
print(sorted_features)

