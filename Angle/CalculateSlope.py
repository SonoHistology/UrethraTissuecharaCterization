#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:53:43 2023

@author: haoweitai
"""
import cv2
import numpy as np 
import ImageGen as ImgGen
import BeamAngleCalc as BAng
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calculate_slope(mask, focal_point, ax2_xll, ax2_xul, ax2_yll, ax2_yul):
    """Calculate the slope of each non-zero pixel with respect to a focal point."""
    
    mask[mask < 100] = 0
    mask[mask > 200] = 255
    
    # Get the height and width of the mask (and the image)
    height, width = mask.shape

    # Initialize an array for the slopes
    slopes = np.zeros_like(mask, dtype=float)
    
    # Initialize an array for the deltas (changes in x and y coordinates)
    deltas = np.zeros_like(mask, dtype=(float, 2))
    
    # Create evenly spaced vectors for x and y coordinates based on the provided limits
    y_vector = np.linspace(ax2_yll, ax2_yul, num = height)
    x_vector = np.linspace(ax2_xll, ax2_xul, num = width)
    
    # Get the focal point coordinates and convert them to cm (from the original meters)
    x_b, y_b = focal_point
    x_b = x_b * 100
    y_b = y_b * 100
        
    # Calculate the slope for each non-zero pixel
    for y in range(height):
        for x in range(width):
            # Get the pixel coordinates and normalize to actual size
            x_pixel_normalized = x_vector[x]
            y_pixel_normalized = y_vector[y]

            # Calculate the slope and store in the slopes array
            # Avoid division by zero by checking if x_pixel_normalized != x_b
            if x_pixel_normalized != x_b:
                delta_y = y_b - y_pixel_normalized
                delta_x = x_b - x_pixel_normalized
                slope = delta_y / delta_x
                slopes[y, x] = slope
                deltas[y, x] = (delta_y, delta_x)
    
    # Convert mask to boolean values (True for non-zero, False for zero)
    mask_bool = mask.astype(bool)
    
    # Extract the delta values for y and x
    delta_y = deltas[..., 0]
    delta_x = deltas[..., 1]
    
    # Calculate slope in degrees using arctan2 (which gives the angle in radians) and convert it to degrees
    # We take absolute values to avoid negative degrees
    slopes_degrees = np.abs(np.degrees(np.arctan2(delta_y, delta_x)))
    
    # Create an empty array (all zeros) of the same shape as slopes_degrees
    result = np.zeros_like(slopes_degrees)

    # Apply the boolean mask to the slopes_degrees. This will retain the slope degrees where mask is True (non-zero) and put zeros where mask is False.
    result[mask_bool] = slopes_degrees[mask_bool]
    
    return result

def main():
    # Here you can put your mask image.
    pname = ('./DICOM/')
    fname = '134742_N58DSA1G'
    i_frame = 0
    
    # Here you can put your mask image.
    mask = cv2.imread('./Img/New/sub5/img22.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Calculate the slopes in degrees
    (intersection_x_adjusted, intersection_y_original, 
         angle_red_line_degrees, angle_blue_line_degrees) = BAng.BeamAngleCalc(mask)
    
    # Convert spatial variables from meters to centimeters for later calculations
    ax2_yll = 0 # start depth in cm
    ax2_yul = 7.99 # end depth in cm
    
    # Calculate height in actual units
    height_actual_units = ax2_yul - ax2_yll
    
    # Calculate height in pixels
    height_pixels = mask.shape[0]

    # Calculate width in pixels
    width_pixels = mask.shape[1]

    # Calculate width in actual units using the formula
    width_actual_units = (height_actual_units / height_pixels) * width_pixels
    
    # Since the middle of the width is 0, we can calculate the left and right bounds
    ax2_xll = -width_actual_units / 2  # Left bound (negative)
    ax2_xul = width_actual_units / 2   # Right bound (positive)
    
    # Convert focal point from pixel coordinates to actual units
    coordinate_x = intersection_x_adjusted * (width_actual_units / width_pixels) + ax2_xll
    coordinate_y = intersection_y_original * (height_actual_units / height_pixels) + ax2_yll

    focal_point = (coordinate_x, coordinate_y)

    slopes_degrees = calculate_slope(mask, focal_point, ax2_xll, ax2_xul, ax2_yll, ax2_yul)
    # you can add more function calls or any other code here for debugging.

    # raw b-mode image
    height, width = mask.shape
    
    # slopes
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax = fig.add_subplot()
    im = ax.imshow(slopes_degrees, cmap='jet',
                   extent=[ax2_xll, ax2_xul, ax2_yul, ax2_yll])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)', color='red')
    ax.set_ylabel('y (cm)', color='red')
    ax.set_title('Pixel Slope Angle (to Y-axis)')
    plt.show()
    return slopes_degrees

if __name__ == "__main__":
    slopes_degrees = main()