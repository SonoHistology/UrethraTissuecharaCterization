#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jul 10 17:55:22 2023
@author: haoweitai
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

def perform_curve_fitting(x, y):
    """
    Performs linear regression on the given x and y data.
    Returns the slope and p-value of the regression.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value

def calculate_row_statistics(row, apply_CI=False):
    """
    Calculates mean, SD, and 95% CI for each row in a DataFrame.
    :param row: A row of DataFrame
    :param apply_CI: A boolean, if True, mean is calculated for data within 95% CI, else mean is calculated for all data.
    :return: A Series containing mean, SD, and CI
    """
    original_row = row.dropna()  # Remove any missing values in the row
    mean_original = np.mean(original_row)
    sd_original = np.std(original_row, ddof=1)  # Calculate Standard Deviation
    sem_original = sd_original / np.sqrt(len(original_row))  # Calculate Standard Error of the Mean
    ci_original = sem_original * stats.t.ppf((1 + 0.95) / 2., len(original_row) - 1)  # Calculate 95% Confidence Interval
    
    if apply_CI:
        ci_upper = mean_original + ci_original
        ci_lower = mean_original - ci_original
        row = row[(row <= ci_upper) & (row >= ci_lower)]
        mean = np.mean(row)
        sd = np.std(row, ddof=1) if len(row) > 1 else np.nan  # Avoid division by zero error
        sem = sd / np.sqrt(len(row)) if len(row) > 1 else np.nan  # Avoid division by zero error
        ci = sem * stats.t.ppf((1 + 0.95) / 2., len(row) - 1) if len(row) > 1 else np.nan  # Avoid division by zero error
        return pd.Series({'Mean_CI': mean, 'SD_CI': sd, 'CI': ci})
    else:
        mean = mean_original
        sd = sd_original
        return pd.Series({'Mean': mean, 'SD': sd})

def process_subfolders(parent_path, use_global_max_cutoff=False):
    """
    Iterates over all subfolders and processes excel files found in each subfolder.
    """
    global_max_index = None  # Initialize global_max_index
    global_min_index = None  # Initialize global_min_index

    for pname in os.listdir(parent_path):
        # Skip non-directory files
        if not os.path.isdir(os.path.join(parent_path, pname)) or pname == '.DS_Store':
            continue
        
        sub_parent_path = os.path.join(parent_path, pname)
        subfolders = [f.path for f in os.scandir(sub_parent_path) if f.is_dir()]
        
        slopes_dict = {}  # To store the slopes for each subfolder
        p_values_dict = {}  # Initialize the p-values dictionary
        
        # First, find the global max index from the 'image' subfolder, if applicable
        if use_global_max_cutoff:
            # First, find the global max index from the 'image' subfolder, if applicable
            image_subfolder_path = os.path.join(sub_parent_path, 'image')
            if os.path.isdir(image_subfolder_path):
                global_max_index, global_min_index = process_files_in_subfolder(image_subfolder_path, slopes_dict, p_values_dict, sub_parent_path, True)
        
            # Now process all subfolders, using the global cutoffs
            for subfolder in subfolders:
                if os.path.basename(subfolder) != 'image':  # Skip 'image' subfolder since it's already processed
                    process_files_in_subfolder(subfolder, slopes_dict, p_values_dict, sub_parent_path, True, global_max_index, global_min_index)
        else:
            # Process all subfolders without any global cutoffs
            for subfolder in subfolders:
                process_files_in_subfolder(subfolder, slopes_dict, p_values_dict, sub_parent_path, False)

        # Save the slopes to an Excel file
        slopes_df = pd.DataFrame(slopes_dict, index=[pname])
        slopes_df.to_excel(os.path.join(sub_parent_path, 'slopes.xlsx'))
        
        # Save the p-values to a separate Excel file
        p_values_df = pd.DataFrame(p_values_dict, index=[pname])
        p_values_df.to_excel(os.path.join(sub_parent_path, 'p_values.xlsx'))
        
def process_files_in_subfolder(subfolder, slopes_dict, p_values_dict, sub_parent_path, use_global_max_cutoff=False, global_max_index=None, global_min_index=None):
    """
    Processes all excel files in the given subfolder and performs plotting and saving operations.
    Applies a cutoff to the data if a global max index is provided.
    """
    subfolder_name = os.path.basename(subfolder)
    files = [file for file in os.listdir(subfolder) if file.endswith('.xlsx') or file.endswith('.xls')]
    df_list = []

    # Get a list of file paths in the subfolder and sort them
    # This example sorts the files by name in ascending order
    file_paths = sorted([os.path.join(subfolder, file) for file in files])
    
    # Loop over the sorted file paths
    for file_path in file_paths:
        # Read the excel file
        df = pd.read_excel(file_path, engine='openpyxl')
        # Append the 'mean' column to the list
        df_list.append(df['mean'])

    if not df_list:  # Skip if there are no excel files in the subfolder
        return

    all_data = pd.concat(df_list, axis=1)
    apply_CI = False  # Set to True if Confidence Interval should be applied
    statistics_df = all_data.apply(lambda row: calculate_row_statistics(row, apply_CI), axis=1)

    # Apply the global max and min cutoff if this subfolder is 'image' or if a global max index is provided
    mean_column = 'Mean_CI' if apply_CI else 'Mean'
    if use_global_max_cutoff and subfolder_name == 'image':
        # First find the global max index starting from the 20th index onwards
        max_index = statistics_df[mean_column][20:].idxmax()
        global_max_index = max_index  # Apply a buffer after the max index
    
        # Now find the global min index within the range from 0 to global_max_index
        # Note: If max_index is the last index, this will result in an empty slice; handle accordingly
        if max_index < len(statistics_df) - 1:
            min_index = statistics_df[mean_column][:global_max_index].idxmin()
        else:
            min_index = statistics_df[mean_column].idxmin()
    
        global_min_index = min_index

    if global_max_index is not None and global_min_index is not None:
        # Apply cutoff from global min to global max + 5
        statistics_df = statistics_df.loc[global_min_index:global_max_index]
        all_data = all_data.loc[global_min_index:global_max_index]
    
    if statistics_df[mean_column].isna().all():  # Skip if there is no valid data to plot
        return
    
    valid_statistics_df = statistics_df.dropna(subset=[mean_column])
    valid_statistics_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinity with NaN
    valid_statistics_df.dropna(subset=[mean_column], inplace=True)  # Drop NaN rows
    
    # Perform curve fitting for the entire data
    slope, intercept, r_value, p_value = perform_curve_fitting(valid_statistics_df.index, valid_statistics_df[mean_column])
    slopes_dict[subfolder_name] = slope
    p_values_dict[subfolder_name] = p_value

    # Perform curve fitting for the last third of the data
    last_third_index = len(valid_statistics_df) // 7
    x_last_third = valid_statistics_df.index[-last_third_index:]
    y_last_third = valid_statistics_df[mean_column][-last_third_index:]
    slope_last_third, intercept_last_third, r_value_last_third, p_value_last_third = perform_curve_fitting(x_last_third, y_last_third)
    slopes_dict[subfolder_name + '_last_fifth'] = slope_last_third
    p_values_dict[subfolder_name + '_last_fifth'] = p_value_last_third
    
    # Calculate R-squared value
    r_squared = r_value**2
    
    plot_and_save(statistics_df, slope, intercept, r_squared, p_value, sub_parent_path, subfolder, apply_CI)
    # statistics_df[[mean_column]].to_excel(os.path.join(sub_parent_path, subfolder.split('/')[-1] + '_row_means.xlsx'), index=False)
    statistics_df.to_excel(os.path.join(sub_parent_path, subfolder.split('/')[-1] + '_row_means.xlsx'), index=False)
    
    # For the 'image' subfolder, return the global max index to be used for other subfolders
    if use_global_max_cutoff and subfolder_name == 'image':
        return global_max_index, global_min_index  # Return both indices

def plot_and_save(statistics_df, slope, intercept, r_squared, p_value, sub_parent_path, subfolder, apply_CI=False):
    """
    Plots the data and saves it to a tiff file.
    :param statistics_df: DataFrame containing the statistics.
    :param slope: Slope of the fitted line.
    :param intercept: Intercept of the fitted line.
    :param r_squared: R-squared value of the fit.
    :param p_value: P-value of the fit.
    :param sub_parent_path: Path to save the plot.
    :param subfolder: Name of the subfolder.
    :param apply_CI: Boolean indicating whether to apply 95% Confidence Interval or not.
    """
    plt.figure(figsize=(8, 8))
    mean_column = 'Mean_CI' if apply_CI else 'Mean'
    
    if apply_CI:
        sns.lineplot(x=statistics_df.index, y=mean_column, data=statistics_df, color='black', linewidth=2.5)
        plt.fill_between(statistics_df.index, statistics_df[mean_column] - statistics_df['SD_CI'], statistics_df[mean_column] + statistics_df['SD_CI'], color='grey', alpha=0.2)
    else:
        sns.lineplot(x=statistics_df.index, y=mean_column, data=statistics_df, color='black', linewidth=2.5)
        plt.fill_between(statistics_df.index, statistics_df[mean_column] - statistics_df['SD'], statistics_df[mean_column] + statistics_df['SD'], color='grey', alpha=0.2)
    
    # Plot the linear regression line
    plt.plot(statistics_df.index, slope * statistics_df.index + intercept, color='red', label=f'Slope: {slope:.2f}\n$R^2$: {r_squared:.2f}\nP-Value: {p_value:.2e}')

    plt.xlabel('Angle', fontsize=24)
    plt.ylabel('Mean', fontsize=24)
    plt.xlim([0, 90])
    plt.xticks([0, 30, 60, 90], fontsize=24)

    ymin, ymax = plt.ylim()
    y_ticks = np.linspace(ymin, ymax, 5)
    plt.yticks(y_ticks, [f"{tick:.2f}" for tick in y_ticks], fontsize=24)

    # plt.legend()
    plt.tight_layout()

    filename = subfolder.split('/')[-1] + ('_CI_plot.tiff' if apply_CI else '_plot.tiff')
    plt.savefig(os.path.join(sub_parent_path, filename), format='tiff')
    # plt.show()
    plt.close()

if __name__ == "__main__":
    parent_path = './result/'  # Parent directory where your subfolders are stored
    process_subfolders(parent_path, use_global_max_cutoff=True)  # Here, we turn on the use_global_max_cutoff feature