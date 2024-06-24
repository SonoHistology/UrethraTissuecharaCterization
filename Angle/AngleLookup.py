#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:49:40 2023

@author: haoweitai
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.morphology import skeletonize
import emd 
from scipy.interpolate import splprep, splev
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Function to return a polynomial of a chosen degree
def make_poly_func(degree):
    def poly_func(x, *params):
        return sum([p * x**i for i, p in enumerate(reversed(params))])
    return poly_func

def show_image_with_colorbar(image, title):
    """Display the image with a color bar."""
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('jet')
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    plt.axis('off')
    fig.colorbar(im, ax=ax)
    plt.show()

# Define a function to compute the loss for a given degree
def compute_loss(x, y, degree):
    poly_coeff = np.polyfit(x, y, degree)
    poly_fn = np.poly1d(poly_coeff)
    y_pred = poly_fn(x)
    return np.sum((y - y_pred) ** 2)

def AngleLookup(mask):
    """
    This function accepts a mask, finds the largest contour in the mask, 
    computes a minimum area rectangle around this contour and finds its orientation.
    It then applies the orientation value to the mask and returns the updated mask.

    Parameters:
    mask (numpy.ndarray): Input binary mask image.
    
    Returns:
    mask (numpy.ndarray): Mask with applied orientation.
    """
    
    # Thresholding
    mask[mask < 150] = 0
    mask[mask > 200] = 255

    # Convert to binary image
    img = img_as_ubyte(mask)

    # Skeletonize
    thinned = skeletonize(img, method='lee')

    # Get the x, y coordinates of non-zero pixels
    y, x = np.nonzero(thinned)
    
    # B-spline fitting
    tck, u = splprep([x, y], s=30)
    x_new, y_new = splev(np.linspace(0, 1, 1000), tck)
    
    # Derive the B-spline for tangent
    dx_new, dy_new = splev(np.linspace(0, 1, 1000), tck, der=1)

    # Compute the tangent angle (in radians) using arctan2 function
    tangent_angles_rad = np.arctan2(dy_new, dx_new)
       
    # Convert the tangent angles from radians to degrees
    tangent_angles_deg = np.degrees(tangent_angles_rad)
        
    # Map the angles to [0, 180]
    # tangent_angles_deg = (270 - tangent_angles_deg) % 180
    
    # imfs = emd.sift.sift(tangent_angles_deg)
    # tangent_angles_deg = imfs[:, -1]

    # Calculate the smoothed tangent vectors
    dx_smooth = np.cos(tangent_angles_rad)
    dy_smooth = np.sin(tangent_angles_rad)
    
    # Normalize the smoothed tangent vectors
    tangent_vectors_smooth = np.stack([dx_smooth, dy_smooth], axis=1)
    tangent_vectors_smooth = tangent_vectors_smooth / np.linalg.norm(tangent_vectors_smooth, axis=1, keepdims=True)
    
    # Normalize the smoothed tangent vectors in-place
    np.divide(tangent_vectors_smooth, np.linalg.norm(tangent_vectors_smooth, axis=1, keepdims=True), out=tangent_vectors_smooth)
    
    imfs = emd.sift.sift(tangent_angles_rad)
    tangent_angles_rad_smooth = imfs[:,-1]
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.plot(x_new, y_new, 'r-', linewidth=2)  # Plot the B-spline fitted line in red
    plt.title('B-spline Fitting on Skeletonized Data')
    plt.show()

    # Calculate the smoothed tangent vectors
    dx_smooth = np.cos(tangent_angles_rad_smooth)
    dy_smooth = np.sin(tangent_angles_rad_smooth)
    
    # Normalize the smoothed tangent vectors
    tangent_vectors_smooth = np.stack([dx_smooth, dy_smooth], axis=1)
    tangent_vectors_smooth = tangent_vectors_smooth / np.linalg.norm(tangent_vectors_smooth, axis=1, keepdims=True)
    
    # Normalize the smoothed tangent vectors in-place
    np.divide(tangent_vectors_smooth, np.linalg.norm(tangent_vectors_smooth, axis=1, keepdims=True), out=tangent_vectors_smooth)
    
    # Display the smoothed boundary, tangent vectors, and vertical lines
    mask_copy = mask.copy()

    # Initialize an empty list to store line masks
    line_masks = []
    angles = []
    
    for i in range(0, len(x_new), 1):  # Change this number to adjust the spacing of the lines
        x, y = x_new[i], y_new[i]
        tx, ty = tangent_vectors_smooth[i] * 10  # Scale the tangent vector for visualization

        # Calculate the normal vector (orthogonal to the tangent vector)
        normal_vector = np.array([-ty, tx])

        # Calculate the start and end points for the vertical line
        line_length = 10
        start_point = (int(x - line_length * normal_vector[0]), int(y - line_length * normal_vector[1]))
        end_point = (int(x + line_length * normal_vector[0]), int(y + line_length * normal_vector[1]))

        # Create a mask with the red line
        line_mask = np.zeros_like(mask_copy)
        cv2.line(line_mask, start_point, end_point, 255, 1)
        line_masks.append(line_mask)
        angles.append(tangent_angles_deg[i])
    
    # Combine line_masks and angles into arrays
    line_masks = np.stack(line_masks, axis=0)
    angles = np.array(angles)

    # Create an empty array for storing angle values at line locations
    angle_array = np.zeros_like(mask_copy)

    # Assign angle values to the locations of the lines in the new array
    for i in range(len(line_masks)):
        line_mask = line_masks[i]
        angle = angles[i]
        line_locations = np.column_stack(np.where(line_mask > 0))
        for (y, x) in line_locations:
            angle_array[y, x] = angle
    
    # Convert mask to boolean values (True for non-zero, False for zero)
    mask_bool = mask.astype(bool)
    
    # Create an empty array (all zeros) of the same shape as slopes_degrees
    intersection = np.zeros_like(mask)
    
    # Apply the boolean mask to the slopes_degrees. This will retain the slope degrees where mask is True (non-zero) and put zeros where mask is False.
    intersection[mask_bool] = angle_array[mask_bool]
    
    # Apply a Gaussian or median filter to smooth the display
    # smoothed_mask = cv2.GaussianBlur(intersection, (5, 5), 0)  # Use a kernel size of 5x5 and sigma=0 for Gaussian filter
    intersection = cv2.medianBlur(intersection, 7)
    intersection[intersection == 255] = 0
    
    # Find the largest contour in the intersection
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest object
    largest_object_mask = np.zeros_like(mask_copy)
    cv2.drawContours(largest_object_mask, [largest_contour], -1, 255, -1)
    
    # Reserve the largest object in the image along with its pixel intensity
    cleaned_mask = cv2.bitwise_and(intersection, largest_object_mask)
    
    return cleaned_mask

def main():
    """
    Main function.
    """
    # Load the mask image
    mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

    # Find the rotation angle
    cleaned_mask = AngleLookup(mask)

    # Display the result
    show_image_with_colorbar(cleaned_mask, "Cleaned Mask")

    return cleaned_mask

if __name__ == "__main__":
    cleaned_mask = main()