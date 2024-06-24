import os
import glob
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
import ImageGen as ImgGen

import AngleLookup as AngL
from scipy.io import savemat

import CalculateSlope as Slope
import BeamAngleCalc as BAng

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
    # diff_matrix = diff_matrix.astype(np.uint8)
    
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

# Set up initial parameters
pname = './Img/New/'
dname = './Img/Dicom/'
useDiParam = False

for fname in os.listdir(pname):
    # Ensure fname is indeed a directory and not a file like .DS_Store
    directory_path = os.path.join(pname, fname)
    if not os.path.isdir(directory_path) or fname.startswith('.'):
        continue
    
    try:
        # Generate the image and related parameters
        img_params = ImgGen.ImageGen(dname, fname)
        if img_params:
            (image_copy, origo_b, lat_end_b, ax_end_b, ax_st_b, R_b, lat_st_b) = img_params
            useDiParam = True
        else:
            # If img_params is None or empty, consider it as a failed attempt to get parameters
            print(f"No image parameters returned for {fname}.")
            useDiParam = False
            
    except Exception as e:
        print(f"An error occurred while generating image for {fname}: {e}")
        useDiParam = False
    
    # iterate over all .jpg files in the directory
    for filename in glob.glob(os.path.join(pname + fname + '/', '*.jpg')):
        
        # Split the file path into a directory and a file name
        directory, file_name = os.path.split(filename)
        
        image_copy = cv2.imread(pname + fname + '/' + file_name, cv2.IMREAD_GRAYSCALE)
        
        # Normalize the image
        image_copy = cv2.normalize(image_copy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
        # Handle case where image could not be loaded
        if image_copy is None:
            print("Error loading image.")
            exit()
    
        # Generate the mask for the image
        mask = cv2.imread('./Seg/' + fname + '/' + file_name, cv2.IMREAD_GRAYSCALE)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # If no contours were found, skip to next file
        if len(contours) == 0:
            print(f"No contours found in mask for {fname}. Skipping this file.")
            continue

        # Handle case where mask could not be created
        if mask is None:
            print("Error creating mask.")
            exit()
        
        if useDiParam:
        # Convert spatial variables from meters to centimeters for later calculations
            ax2_xll = min(lat_end_b) * 100
            ax2_xul = max(lat_end_b) * 100
            ax2_yll = (min(ax_st_b) - R_b) * 100
            ax2_yul = (max(ax_end_b) - R_b) * 100
            
            focal_point = (origo_b[0], origo_b[1])
        
        else:
            # Calculate the slopes in degrees
            (intersection_x_adjusted, intersection_y_original, 
                 angle_red_line_degrees, angle_blue_line_degrees) = BAng.BeamAngleCalc(image_copy)
            
            ax2_yll = 0 # start depth in cm
            ax2_yul = 7.99 # end depth in cm
            
            # Calculate height in actual units
            height_actual_units = ax2_yul - ax2_yll
            
            # Calculate height in pixels
            height_pixels = image_copy.shape[0]
    
            # Calculate width in pixels
            width_pixels = image_copy.shape[1]
    
            # Calculate width in actual units using the formula
            width_actual_units = (height_actual_units / height_pixels) * width_pixels
            
            # Since the middle of the width is 0, we can calculate the left and right bounds
            ax2_xll = -width_actual_units / 2  # Left bound (negative)
            ax2_xul = width_actual_units / 2   # Right bound (positive)
            
            # Convert focal point from pixel coordinates to actual units
            coordinate_x = intersection_x_adjusted * (width_actual_units / width_pixels) + ax2_xll
            coordinate_y = intersection_y_original * (height_actual_units / height_pixels) + ax2_yll
    
            focal_point = (coordinate_x, coordinate_y)
        
        slopes_degrees = Slope.calculate_slope(mask, focal_point, ax2_xll, ax2_xul, ax2_yll, ax2_yul)
        
        # Clean the mask
        cleaned_mask = AngL.AngleLookup(mask)
        
        # # Create a Gaussian kernel for further image processing
        # kernel_size = 3  # Must be an odd number
        # sigma = 1  # Standard deviation of the Gaussian distribution
        # gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
        # gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # Create a 2D kernel from the 1D kernel
    
        # # Apply the Gaussian kernel to the image
        # cleaned_mask = cv2.filter2D(cleaned_mask, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)
    
        # Handle case where mask could not be cleaned
        if cleaned_mask is None:
            print("Error cleaning mask.")
            exit()
    
        # Calculate the angle differences
        resultant_mask = calculate_tangent_to_focal_point_angle(cleaned_mask, slopes_degrees)
        resultant_mask[np.isnan(resultant_mask)] = 0
    
        # Define a binary erosion kernel
        kernel = np.ones((5, 5), np.uint8)
    
        # Erode the mask and then use it to clean the masks
        eroded_mask = cv2.erode(mask, kernel, iterations = 1)
        cleaned_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=eroded_mask)
        resultant_mask = cv2.bitwise_and(resultant_mask, resultant_mask, mask=eroded_mask)
        
        # Convert zeros in binary_mask to NaN in resultant_mask for visualization
        resultant_mask = resultant_mask.astype(float)
        threshold = 0.1  # or some small value
        resultant_mask[np.abs(cleaned_mask) < threshold] = np.nan

        # Construct the path to save the .mat file
        # We'll use the file_name without the .jpg extension
        mat_file_name = file_name.split('.')[0] + '.mat'
        
        # Define the full path to save the .mat file
        # Assuming you want to save it in the same directory as the image
        mat_file_path = os.path.join(mat_file_name)
        
        # Save the resultant_mask to a .mat file
        savemat(mat_file_path, {'resultant_mask': resultant_mask})