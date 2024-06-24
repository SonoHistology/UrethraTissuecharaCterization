#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:51:08 2023

@author: haoweitai
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def get_bounding_boxes(binary_image):
    """
    Get bounding boxes of the contours in the binary image.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours]


def get_boundaries(cropped_region):
    """
    Get left and right boundaries within the cropped region.
    """
    left_boundary_cropped = np.argmax(cropped_region, axis=1)
    right_boundary_cropped = cropped_region.shape[1] - np.argmax(np.fliplr(cropped_region), axis=1) - 1
    return left_boundary_cropped, right_boundary_cropped


def calculate_angle_from_slope(slope):
    """
    Calculate the angle of a line given its slope, with respect to the negative x-axis.
    """
    angle_radians = np.arctan(-slope)
    angle_degrees = np.degrees(angle_radians)
    if slope < 0:
        angle_degrees += 90
    elif slope > 0:
        angle_degrees += 270
    else:
        angle_degrees = 0
    return angle_degrees


def BeamAngleCalc(image):
    """
    Calculate the angle of beams in the image.
    """
    _, binary_image = cv2.threshold(image, 1, 10, cv2.THRESH_BINARY)
    
    x1, y1, x2, y2 = max(get_bounding_boxes(binary_image), key=lambda x: x[2] * x[3])
    cropped_region = binary_image[y1:y2, x1:x2]
    
    left_boundary_cropped, right_boundary_cropped = get_boundaries(cropped_region)
    left_boundary_final = left_boundary_cropped + x1
    right_boundary_final = right_boundary_cropped + x1

    top_half_y = y1 + (y2 - y1) // 2
    adjusted_y_values = np.arange(0, top_half_y - y1)
    
    left_slope, left_intercept, _, _, _ = linregress(adjusted_y_values, left_boundary_final[:top_half_y - y1])
    right_slope, right_intercept, _, _, _ = linregress(adjusted_y_values, right_boundary_final[:top_half_y - y1])
    
    intersection_y_adjusted = (right_intercept - left_intercept) / (left_slope - right_slope)
    intersection_x_adjusted = left_slope * intersection_y_adjusted + left_intercept
    intersection_y_original = intersection_y_adjusted + y1

    plt.imshow(binary_image, cmap='gray')
    plt.plot(left_boundary_final[:top_half_y - y1], range(y1, top_half_y), color='red', label='Left Boundary')
    plt.plot(right_boundary_final[:top_half_y - y1], range(y1, top_half_y), color='blue', label='Right Boundary')
    plt.scatter([intersection_x_adjusted], [intersection_y_original], color='black', label='Intersection Point')
    plt.legend()
    plt.show()
    
    angle_red_line_degrees = calculate_angle_from_slope(left_slope)
    angle_blue_line_degrees = calculate_angle_from_slope(right_slope)
    
    return intersection_x_adjusted, intersection_y_original, angle_red_line_degrees, angle_blue_line_degrees


def main():
    image_path = "./Img/Dicom/sub14.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return BeamAngleCalc(image)


if __name__ == "__main__":
    intersection_x_adjusted, intersection_y_original, angle_red_line_degrees, angle_blue_line_degrees = main()