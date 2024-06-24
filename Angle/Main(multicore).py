#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:31:46 2023

@author: hatai
"""

import os
import glob
import cv2
import numpy as np 
import pandas as pd
from PIL import Image 
import matplotlib.pyplot as plt
from skimage import filters
import concurrent.futures

# Local imports
import ImageGen as ImgGen
import ImageSeg as ImgSeg
import AngleLookup as AngL
import ImageProcess as ImgPro
import CalculateSlope as Slope
import BeamAngleCalc as BAng

import traceback

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Function to calculate the angle difference matrix
def calculate_tangent_to_focal_point_angle(cleaned_mask, focal_point_angles_deg):
    """
    Calculate the angle between the tangent line of a pixel and 
    the line from each pixel to the focal point.
    """
    # Identify non-zero pixels in the cleaned_mask
    non_zero_indices = cleaned_mask > 0
    
    # Calculate the absolute angle difference for the non-zero pixels
    angle_difference = np.abs(cleaned_mask[non_zero_indices] - focal_point_angles_deg[non_zero_indices])
    
    # Normalize the angle difference to be within 0 to 180 degrees
    angle_difference %= 180
    
    # If the angle difference exceeds 90 degrees, convert it to its smaller complementary angle
    angle_difference[angle_difference > 90] = 180 - angle_difference[angle_difference > 90]
    
    # Initialize a matrix of zeros with the same shape as the cleaned_mask
    diff_matrix = np.zeros_like(cleaned_mask, dtype=float)
    
    # Assign the computed angle differences to the corresponding pixels in the diff_matrix
    diff_matrix[non_zero_indices] = angle_difference
    
    # Convert diff_matrix to 8-bit unsigned integer type for compatibility with OpenCV functions
    diff_matrix = diff_matrix.astype(np.uint8)
    
    return diff_matrix

# Function to plot angle intensity histogram
def plot_angle_intensity_histogram(fname, mask, resultant_mask, original_image, file_prefix, image_type, angle_range=None, bin_size=1):
    """
    Generates a histogram to depict average intensity per angle for a specific image type.
    
    Args:
        mask (numpy.ndarray): 2D array representing the binary mask of the region of interest.
        resultant_mask (numpy.ndarray): 2D array representing the angle of each pixel after processing.
        original_image (numpy.ndarray): 2D array representing the original grayscale image.
        file_prefix (str): Prefix for the output file.
        image_type (str): Type of the image. This will be added to the filename and the plot title.
        angle_range (tuple of float, optional): The angle range to consider. Should be in the form (start, end).
        bin_size (float, optional): The size of each bin in the histogram. Default is 1 degree.

    Outputs:
        A bar plot of the average intensity per angle bin, with error bars representing the standard deviation.
        An Excel file "<file_prefix>_<image_type>_stats_per_angle.xlsx" containing the statistical data used to generate the histogram.
    """

    assert mask.ndim == 2, "Mask should be a 2D array"
    assert resultant_mask.ndim == 2, "Resultant mask should be a 2D array"
    assert original_image.ndim == 2, "Original image should be a 2D array"
    if angle_range is not None:
        assert len(angle_range) == 2 and 0 <= angle_range[0] < angle_range[1] <= 180, "Invalid angle range"

    # Flatten the mask and the original image
    flat = mask.flatten()
    flat_mask = resultant_mask.flatten()
    flat_image = original_image.flatten()

    # Exclude zero pixels
    nonzero_indices = flat.nonzero()
    flat_mask_nonzero = flat_mask[nonzero_indices]
    flat_image_nonzero = flat_image[nonzero_indices]

    # Group angles into bins based on bin size
    bins = np.arange(0, 180 + bin_size, bin_size)  # +bin_size to ensure the last bin is included
    bin_midpoints = (bins[:-1] + bins[1:]) / 2  # Label each bin with its midpoint

    # Assign each non-zero pixel to an angle bin
    flat_mask_binned = pd.cut(flat_mask_nonzero, bins=bins, labels=bin_midpoints)

    # Create a DataFrame for easier manipulation
    data = {'Angle': flat_mask_binned, 'Intensity': flat_image_nonzero}
    df = pd.DataFrame(data)

    # Calculate the average intensity and standard deviation within each angle bin
    stats_per_angle = df.groupby('Angle')['Intensity'].agg(['mean', 'std'])

    # If a specific angle range is provided, filter the data to include only the angles within this range
    if angle_range is not None:
        start_angle, end_angle = angle_range
        stats_per_angle = stats_per_angle.reset_index()
        stats_per_angle = stats_per_angle.loc[start_angle:end_angle]
    
    # Create a directory for the image type if it doesn't exist
    directory = os.path.join(os.getcwd(), fname, image_type)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save stats_per_angle to an Excel file in the image_type directory inside fname directory
    output_file = os.path.join(directory, f"{file_prefix}_stats_per_angle.xlsx")
    stats_per_angle.to_excel(output_file)

    # # Create a bar plot
    # stats_per_angle['mean'].plot(kind='bar', yerr=stats_per_angle['std'], figsize=(10, 7), width=0.8, 
    #                              color='lightgray', edgecolor='black')
    # plt.title(f'Average Intensity Per Angle for {image_type}')
    # plt.ylabel('Average Intensity')
    # plt.show()

# Global variables to keep track of the last processed subfolder and cached parameters
last_subfolder_name = None
cached_params = None

def process_image(filename, dname):
    """
    Process an image given a filename and dname.

    Args:
        filename (str): Path to the image file to process.
        dname (str): Path to the DICOM images directory.

    Returns:
        None
    """
    global last_subfolder_name, cached_params
    useDiParam = False  # Flag to determine whether DICOM parameters are used

    # Extract subfolder name and file name from the filename
    subfolder_name = os.path.basename(os.path.dirname(filename))
    file_name = os.path.basename(filename)

    # Check if the subfolder has changed since the last processed file
    if subfolder_name != last_subfolder_name:
        # Path assuming the DICOM file has no extension
        dicom_path = os.path.join(dname, subfolder_name)
    
        try:
            # Try to use the DICOM directory first
            if os.path.exists(dicom_path):  # Changed to check file existence instead of directory
                img_params = ImgGen.ImageGen(dname, subfolder_name)
                if img_params:
                    cached_params = img_params
                    useDiParam = True  # Set flag only if DICOM parameters are used
                else:
                    print(f"Image parameters could not be obtained for {dicom_path}.")
                    return
            else:
                # If DICOM directory is not found, fall back to JPG file
                jpg_file_path = os.path.join(dname, f"{subfolder_name}.jpg")
                if os.path.exists(jpg_file_path):  # Check if the JPG file exists before trying to read it
                    img_params = BAng.BeamAngleCalc(cv2.imread(jpg_file_path, cv2.IMREAD_GRAYSCALE))
                    if img_params:
                        cached_params = img_params
                        # Don't set useDiParam here since we're using JPG parameters
                    else:
                        print(f"Could not calculate parameters for {subfolder_name}.")
                        return
                else:
                    print(f"JPG file {jpg_file_path} not found. Skipping parameter calculation.")
                    return
        except Exception as e:
            print(f"Error generating image parameters for {subfolder_name}: {e}")
            traceback.print_exc()  # Added traceback to print the stack trace
            return
    
        # Update the last processed subfolder
        last_subfolder_name = subfolder_name

    # Check if parameters are cached; if so, use them for processing
    if not cached_params:
        print(f"No parameters available for {subfolder_name}.")
        return

    # Read the image first as it is required in both cases (useDiParam True or False)
    image_copy = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image_copy is None:
        print(f"Error loading image from {filename}.")
        return
    
    # Normalize the image
    image_copy = cv2.normalize(image_copy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Generate the mask for the image
    mask_filename = os.path.join('Seg', subfolder_name, file_name)
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error creating mask from {mask_filename}.")
        return

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"No contours found in mask for {subfolder_name}. Skipping this file.")
        return

    try:
        if useDiParam:
            # Use DICOM parameters for spatial variables and focal point
            (_, origo_b, lat_end_b, ax_end_b, ax_st_b, R_b, lat_st_b) = cached_params
            ax2_xll = min(lat_end_b) * 100
            ax2_xul = max(lat_end_b) * 100
            ax2_yll = (min(ax_st_b) - R_b) * 100
            ax2_yul = (max(ax_end_b) - R_b) * 100
            focal_point = (origo_b[0][0], origo_b[0][1])
        else:
            # Fallback calculations if DICOM parameters are not used
            (intersection_x_adjusted, intersection_y_original, 
             angle_red_line_degrees, angle_blue_line_degrees) = cached_params
            ax2_yll = 0  # start depth in cm
            ax2_yul = 7.99  # end depth in cm
            
            # Calculate height in actual units
            height_actual_units = ax2_yul - ax2_yll
            
            # Calculate height in pixels
            height_pixels = image_copy.shape[0]
        
            # Calculate width in pixels
            width_pixels = image_copy.shape[1]
        
            # Calculate width in actual units using the formula
            width_actual_units = (height_actual_units / height_pixels) * width_pixels
            
            # Since the middle of the width is 0, calculate the left and right bounds
            ax2_xll = -width_actual_units / 2  # Left bound (negative)
            ax2_xul = width_actual_units / 2   # Right bound (positive)
            
            # Convert focal point from pixel coordinates to actual units
            coordinate_x = intersection_x_adjusted * (width_actual_units / width_pixels) + ax2_xll
            coordinate_y = intersection_y_original * (height_actual_units / height_pixels) + ax2_yll
            
            focal_point = (coordinate_x, coordinate_y)

        # Calculate the slopes in degrees
        slopes_degrees = Slope.calculate_slope(mask, focal_point, ax2_xll, ax2_xul, ax2_yll, ax2_yul)

        # Clean the mask
        cleaned_mask = AngL.AngleLookup(mask)
        if cleaned_mask is None:
            print("Error cleaning mask.")
            return

        # Calculate the angle differences
        resultant_mask = calculate_tangent_to_focal_point_angle(cleaned_mask, slopes_degrees)
        resultant_mask[np.isnan(resultant_mask)] = 0

        # Process the image
        features, features_full = ImgPro.ImageProcess(mask, image_copy)

        # Erode and clean the mask
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        cleaned_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=eroded_mask)
        resultant_mask = cv2.bitwise_and(resultant_mask, resultant_mask, mask=eroded_mask)

        # Convert zeros in binary_mask to NaN in resultant_mask for visualization
        resultant_mask = resultant_mask.astype(float)
        resultant_mask[resultant_mask == 0] = np.nan

        # Plot histograms for features
        for feature_name, feature_matrix in features_full.items():
            plot_angle_intensity_histogram(subfolder_name, mask, resultant_mask, np.squeeze(feature_matrix), file_prefix=file_name, 
                                           image_type=feature_name)

        # Plot histogram for the original image
        plot_angle_intensity_histogram(subfolder_name, mask, resultant_mask, image_copy, file_prefix=file_name, 
                                       image_type='image')

    except Exception as e:
        # Log any exceptions encountered during processing
        error_msg = f"An error occurred while processing image {filename}: {e}"
        print(error_msg)
        traceback.print_exc()

    return None

def process_images_in_directory(directory_path, dname, parallel=False):
    """
    Processes all JPEG images within a given directory.

    Args:
        directory_path (str): Path to the directory containing images.
        dname (str): Path to the DICOM images directory.
        parallel (bool): If True, uses parallel processing. Otherwise, processes images serially.
    """
    if parallel:
        # Parallel processing using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Create a list of futures for all images using a list comprehension
            futures = [executor.submit(process_image, filename, dname)
                       for fname in os.listdir(directory_path) if not fname.startswith('.') and os.path.isdir(os.path.join(directory_path, fname))
                       for filename in glob.glob(os.path.join(directory_path, fname, '*.jpg'))]
            
            # Iterate over the completed futures and handle results
            for future in futures:
                try:
                    # We don't print the result, just retrieve it to catch possible exceptions.
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
    else:
        # Serial processing of images
        for fname in os.listdir(directory_path):
            if fname.startswith('.') or not os.path.isdir(os.path.join(directory_path, fname)):
                continue

            for filename in glob.glob(os.path.join(directory_path, fname, '*.jpg')):
                try:
                    # Directly call the function with dname as well and handle any potential errors
                    process_image(filename, dname)
                except Exception as e:
                    print(f"An error occurred while processing {filename}: {e}")

# Set up initial parameters
pname = './Img/New/'
dname = './Img/Dicom/'  # The DICOM images directory path
parallel_processing = False  # Set to True for parallel processing

# Call the image processing function with the specified parameters
process_images_in_directory(pname, dname, parallel=parallel_processing)
