#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:58:24 2023

@author: haoweitai
"""
import os

import cv2
import pydicom
import numpy as np
import pprint
import kk_dicom_util as util
from datetime import datetime
from scipy import ndimage

def ImageGen(pname,fname):
    pp = pprint.PrettyPrinter(width=120)

    pname = pname
    fname = fname

    roi_cntr = (0.0, 0.04)  # center of elliptical roi w/o considering offset, in m
    roi_a = 0.02  # semi-major axis of the elliptical roi, in m
    roi_b = 0.015  # semi-minor axis of the elliptical roi, in m

    ds = pydicom.dcmread(pname + fname)

    content_time = util.ge_pvt_elem_getter(ds, '#DCMContentTime')
    content_time = datetime.strptime(content_time, '%Y/%m/%d-%H:%M:%S')
    content_time = content_time.strftime('%H:%M:%S')

    # ---------------------- Public Fields ---------------------- #
    # total rows and columns of image stored in public field
    rows = ds[0x0028, 0x0010].value
    cols = ds[0x0028, 0x0011].value

    # pixel data (in the public field), format: (number of frames, rows, cols, rgb)
    pixel_data = ds.pixel_array
    # number of frames in the dicom file
    n_frames = pixel_data.shape[0]
    # b-mode region, dicom object, dataset
    region_b = ds[0x00186011][0]

    # b-mode image stored in public field
    image_b = pixel_data[
        :, region_b.RegionLocationMinY0:region_b.RegionLocationMaxY1,
        region_b.RegionLocationMinX0:region_b.RegionLocationMaxX1, :]

    # calculate the width and height of each region in pixels
    # b-mode and contrast mode images have same width and height
    image_width_pixel_b = (region_b.RegionLocationMaxX1
                           - region_b.RegionLocationMinX0)
    image_height_pixel_b = (region_b.RegionLocationMaxY1
                            - region_b.RegionLocationMinY0)

    region_width_pixel = (region_b.RegionLocationMaxX1 -
                          region_b.RegionLocationMinX0)
    region_height_pixel = (region_b.RegionLocationMaxY1 -
                           region_b.RegionLocationMinY0)
    # find coordinates of upper left and lower right corners of regions
    # wrt to reference pixel
    xll = -region_b.ReferencePixelX0
    yll = -region_b.ReferencePixelY0
    xul = region_width_pixel - region_b.ReferencePixelX0
    yul = region_height_pixel - region_b.ReferencePixelY0
    # convert pixels to cm
    xll *= region_b.PhysicalDeltaX
    yll *= region_b.PhysicalDeltaY
    xul *= region_b.PhysicalDeltaX
    yul *= region_b.PhysicalDeltaY

    # ---------------------- Private fields ---------------------- #
    grid_size_b = util.ge_pvt_elem_getter(ds, 'GridSize')
    origo_b = util.ge_pvt_elem_getter(ds, 'Origo') #coordinate of the virtual focal point
    de_b = util.ge_pvt_elem_getter(ds, 'DepthEnd')
    ds_b = util.ge_pvt_elem_getter(ds, 'DepthStart')
    anti_log_vec_b = util.ge_pvt_elem_getter(
        ds, '#AntiLogLawVector')
    vec_angles_b = util.ge_pvt_elem_getter(
        ds, 'VectorAngles')
    
    if len(vec_angles_b) > 1:
        
        (raw_n_frames_b,
         frame_time_b, 
         raw_img_b, 
         N_b, 
         M_b, 
         R_b, 
         r_b,
         theta_b, 
         lat_st_b, ax_st_b, lat_end_b, ax_end_b) = util.ge_raw_us_img_info_getter(
             ds, grid_size_b[1], 
             origo_b[1],
             de_b[1],
             ds_b[1], 
             vec_angles_b[1])
        

    # scan converted image width and height in cartesian coordinates
    wx_b = lat_end_b.max() - lat_end_b.min()
    wy_b = ax_end_b.max() - ax_st_b.min()

    # Desired converted image width and height in pixels
    nx_b = 720
    ny_b = int(np.round(nx_b * wy_b / wx_b))

    (x_b, y_b, 
     X_b, Y_b) = util.cart_grid_and_roi_maker(
         ds, R_b,
         lat_st_b, ax_st_b, lat_end_b, ax_end_b,
         roi_cntr, roi_a, roi_b, nx_b, ny_b)

    # ---------------------- Illustration ---------------------- #
    # Pick a frame number for illustration purposes
    i_frame = 0
    i_frame_full_img = pixel_data[i_frame, :, :, :]
    i_frame_img_b = image_b[i_frame, ...]

    # raw images
    i_frame_raw_img_b = raw_img_b[i_frame, ...]

    # Get the height and width of the image
    width, height = i_frame_raw_img_b.shape

    # Calculate the middle index of the height
    mid_height = height // 2

    # Slice the image from the top to the middle half
    i_frame_raw_img_b = i_frame_raw_img_b[:, :mid_height]
    i_frame_raw_img_b = ndimage.rotate(i_frame_raw_img_b, -90)
    i_frame_raw_img_b = np.fliplr(i_frame_raw_img_b)
    i_frame_scan_cnvrtd_img_b = util.scan_converter(
        r_b, theta_b, i_frame_raw_img_b, x_b, y_b, 3)

    # Normalize the image_copy array to the range [0, 1]
    image_copy = (i_frame_scan_cnvrtd_img_b - np.min(i_frame_scan_cnvrtd_img_b)) / (np.max(i_frame_scan_cnvrtd_img_b) - np.min(i_frame_scan_cnvrtd_img_b))

    # Convert the image_copy array to uint8 and scale it to the range [0, 255]
    image_copy = (image_copy * 255).astype(np.uint8)
    
    return image_copy, origo_b, lat_end_b, ax_end_b, ax_st_b, R_b, lat_st_b