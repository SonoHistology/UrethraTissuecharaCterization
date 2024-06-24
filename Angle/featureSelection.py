#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:42:53 2023

@author: hatai
"""

# import pandas as pd
# from scipy.stats import linregress
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # Load the Excel file
# file_path = 'slope.xlsx'
# data = pd.read_excel(file_path)

# # Normalizing the data for all columns except 'Age', 'Label', and 'Name'
# scaler = MinMaxScaler(feature_range=(10, 60))
# all_columns_except_age_label_name = data.columns.difference(['Age', 'Label', 'Name'])
# normalized_data = scaler.fit_transform(data[all_columns_except_age_label_name])
# normalized_df = pd.DataFrame(normalized_data, columns=all_columns_except_age_label_name)
# normalized_df['Age'] = data['Age']
# normalized_df['Label'] = data['Label']  # Add 'Label' column

# # Define colors for the labels
# label_colors = {'I': 'blue', 'C': 'red'}

# # Plotting
# plt.rcParams.update({'font.size': 24})

# # Function to plot columns with regression
# def plot_columns(columns, title_suffix):
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#     for i, column in enumerate(columns):
#         ax = axes[i//2, i%2]
#         slope, intercept, r_value, p_value, std_err = linregress(normalized_df['Age'], normalized_df[column])
#         for label, color in label_colors.items():
#             subset = normalized_df[normalized_df['Label'] == label]
#             ax.scatter(subset['Age'], subset[column], color=color, alpha=0.5, label=label, s=100)
#         sns.regplot(x='Age', y=column, data=normalized_df, ax=ax, scatter=False, line_kws={'color': 'black'}, ci=95, truncate=False)
#         ax.set_title(f'Slope: {slope:.2f}')
#         ax.set_xlabel('Age')
#         ax.set_ylabel(column)
#         ax.tick_params(labelsize=20)
#         ax.set_xlim(left=20)
#         ax.set_ylim(0, 90)
#         ax.grid(False)
#         ax.legend()
#     plt.tight_layout()
#     plt.show()

# # Calculating the slope and R^2 value of the linear regression for each normalized column against 'Age'
# r2_scores = {}
# flattest_slopes = {}
# for column in normalized_df.columns[:-2]:  # Excluding 'Age' and 'Label' column
#     slope, intercept, r_value, p_value, std_err = linregress(normalized_df['Age'], normalized_df[column])
#     r2_scores[column] = r_value**2  # Storing the R^2 value
#     flattest_slopes[column] = abs(slope)  # Storing the absolute value of the slope

# # Sorting the R^2 values and slopes
# sorted_r2_scores = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
# sorted_flattest_slopes = sorted(flattest_slopes.items(), key=lambda x: x[1])

# # Get the top 4 columns with the highest R^2 values and the flattest slopes
# highest_r2_columns = [col[0] for col in sorted_r2_scores[:4]]
# flattest_slope_columns = [col[0] for col in sorted_flattest_slopes[:4]]

# # Plotting the top 4 columns with the highest R^2 values and flattest slopes
# plot_columns(highest_r2_columns, 'Highest R^2')
# plot_columns(flattest_slope_columns, 'Flattest Slopes')

import pandas as pd
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Load the Excel file
file_path = 'slope.xlsx'
data = pd.read_excel(file_path)

# Normalizing the data for all columns except 'Age', 'Label', and 'Name'
scaler = MinMaxScaler(feature_range=(10, 60))
all_columns_except_age_label_name = data.columns.difference(['Age', 'Label', 'Name'])
normalized_data = scaler.fit_transform(data[all_columns_except_age_label_name])
normalized_df = pd.DataFrame(normalized_data, columns=all_columns_except_age_label_name)
normalized_df['Age'] = data['Age']
normalized_df['Label'] = data['Label']  # Add 'Label' column

# Define colors for the labels
label_colors = {'I': 'blue', 'C': 'red'}

# Function to plot columns with regression and black boundaries for scatter plots
def plot_columns(columns, title_suffix):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for i, column in enumerate(columns):
        ax = axes[i//2, i%2]
        slope, intercept, r_value, p_value, std_err = linregress(normalized_df['Age'], normalized_df[column])
        for label, color in label_colors.items():
            subset = normalized_df[normalized_df['Label'] == label]
            ax.scatter(subset['Age'], subset[column], color=color, alpha=0.5, label=label, s=100, edgecolor='black')
        sns.regplot(x='Age', y=column, data=normalized_df, ax=ax, scatter=False, line_kws={'color': 'black'}, ci=95, truncate=False)
        ax.set_title(f'{column} (Slope: {slope:.2f})')
        ax.set_xlabel('Age')
        ax.set_ylabel(column)
        ax.tick_params(labelsize=20)
        ax.set_xlim(left=20)
        ax.set_ylim(0, 90)
        ax.grid(False)
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.show()

# Function to perform PCA and plot
def perform_pca_and_plot(columns, title):
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(normalized_df[columns])
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1'])
    pc_df['Age'] = normalized_df['Age']
    pc_df['Label'] = normalized_df['Label']

    plt.figure(figsize=(15, 12))  # Sets the size of the plot

    for label, color in label_colors.items():
        subset = pc_df[pc_df['Label'] == label]
        plt.scatter(subset['Age'], subset['PC1'], color=color, alpha=0.5, label=label, s=300, edgecolor='black')  # s=300 sets the size of scatter points
    sns.regplot(x='Age', y='PC1', data=pc_df, ax=plt.gca(), scatter=False, line_kws={'color': 'black'}, ci=95, truncate=False)
    plt.xlabel('Age', fontsize=30)  # Adjust font size for x-axis label
    plt.ylabel('Calculated Urethra Score', fontsize=30)  # Adjust font size for y-axis label
    # plt.title(title, fontsize=24)  # Adjust font size for title
    plt.tick_params(labelsize=30)  # Adjust font size for tick labels
    plt.grid(False)
    # plt.legend(fontsize=14)  # Uncomment and adjust font size for legend
    plt.tight_layout()
    plt.show()


# Calculating and sorting R^2 values and slopes
r2_scores = {}
flattest_slopes = {}
for column in normalized_df.columns[:-2]:  # Excluding 'Age' and 'Label' column
    slope, intercept, r_value, p_value, std_err = linregress(normalized_df['Age'], normalized_df[column])
    r2_scores[column] = r_value**2
    flattest_slopes[column] = abs(slope)

sorted_r2_scores = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
sorted_flattest_slopes = sorted(flattest_slopes.items(), key=lambda x: x[1])

# Getting the top 4 columns
highest_r2_columns = [col[0] for col in sorted_r2_scores[:4]]
flattest_slope_columns = [col[0] for col in sorted_flattest_slopes[:4]]

# Plotting the top 4 columns and their PCA
plot_columns(highest_r2_columns, 'Highest R^2')
perform_pca_and_plot(highest_r2_columns, 'Highest R^2')

plot_columns(flattest_slope_columns, 'Flattest Slopes')
perform_pca_and_plot(flattest_slope_columns, 'Flattest Slopes')

