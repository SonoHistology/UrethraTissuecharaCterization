#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:53:34 2023

@author: haoweitai
"""
import os
import cv2
import six
import SimpleITK as sitk
import numpy as np 
import radiomics
from radiomics import featureextractor, getFeatureClasses

paramsFile = os.path.abspath(r'Voxel.yaml')

def ensure_3d(img):
    """Ensure that the image is 3D. If it's 2D, add an extra dimension."""
    if len(img.shape) == 2:
        return img[np.newaxis, ...]
    return img

def place_feature_in_mask(mask, largest_contour, feature_matrix):
    """
    Place the feature matrix values in the original mask's location corresponding to the largest contour.

    Args:
    - mask (numpy.ndarray): The original binary mask.
    - largest_contour (list of tuple): List containing the largest contour points.
    - feature_matrix (numpy.ndarray): The matrix containing feature values.

    Returns:
    - numpy.ndarray: A mask with the feature matrix values placed in the largest contour's location.
    """
    # Squeeze out singleton dimensions from the feature matrix
    feature_matrix = np.squeeze(feature_matrix)
    
    # Create a blank mask to store the largest contour
    contour_mask = np.zeros_like(mask, dtype=np.uint8)

    # Draw the largest contour on the contour_mask
    cv2.drawContours(contour_mask, [largest_contour], 0, (255), thickness=cv2.FILLED)

    # Extract the region defined by the largest contour
    ys, xs = np.where(contour_mask > 0)
    extracted_region = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]

    # Resize the feature matrix to match the extracted region's shape
    resized_feature_matrix = cv2.resize(feature_matrix, (extracted_region.shape[1], extracted_region.shape[0]))

    # Initialize an output mask with zeros
    output_mask = np.zeros_like(mask, dtype=resized_feature_matrix.dtype)

    # Place the resized feature matrix values in the output mask based on the contour_mask
    output_mask[ys.min():ys.max()+1, xs.min():xs.max()+1] = resized_feature_matrix

    return output_mask

def ImageProcess(mask,image_n):
    
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are no contours, exit
    if not contours:
        raise ValueError("No contours found in the mask.")
        
        # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask
    clean_mask = np.zeros_like(mask)

    # Fill in the largest contour on the blank mask
    cv2.drawContours(clean_mask, [largest_contour], 0, (255), thickness=cv2.FILLED)

    # Normalize the grayscale image to a specific number of gray levels (e.g., 256)
    normalized_image = (image_n / 255 * (256 - 1)).astype(np.uint8)
    # normalized_image = normalized_image.astype(np.float64)
    # Compute the GLCM with a specific step (e.g., 1) and direction (e.g., 0 degrees)
    # glcm = graycomatrix(normalized_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    featureClasses = getFeatureClasses()
    
    # Convert OpenCV images to SimpleITK images
    image_sitk = sitk.GetImageFromArray(ensure_3d(normalized_image))
    mask_sitk = sitk.GetImageFromArray(ensure_3d(clean_mask))

    featureVector = extractor.execute(image_sitk, mask_sitk, voxelBased=True)
    
    # Dictionary to store feature values
    features = {}

    for featureName, featureValue in six.iteritems(featureVector):
        if isinstance(featureValue, sitk.Image):
            matrix = sitk.GetArrayFromImage(featureValue)
            features[featureName] = matrix
    
    features_full = {}
    # For each feature in the features dictionary, replace its value with the result of place_feature_in_mask
    for feature_name, feature_matrix in features.items():
        features_full[feature_name] = place_feature_in_mask(mask, largest_contour, feature_matrix)

    return features, features_full

def main():
    """
    Main function.
    """
    # Load the mask image
    image = cv2.imread('20.jpg', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Find the rotation angle
    features, features_full = ImageProcess(mask,image)

    return features, features_full

if __name__ == "__main__":
    features, features_full = main()