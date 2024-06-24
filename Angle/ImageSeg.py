#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:08:49 2023

@author: haoweitai
"""
import numpy as np
import torch
import argparse
from PIL import Image
import torch.nn.functional as F
from skimage import measure
from utils.data_loading import BasicDataset
from unet import UNet
import skimage.filters
from skimage import data, color, io, img_as_float
from skimage.morphology import disk  # noqa
from skimage.morphology import (closing) 

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/muscle/checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    #parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    #parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def remove_small_objects(image):
    # Label connected components
    labels = measure.label(image)
    
    # Get properties of labeled regions
    props = measure.regionprops(labels)
    
    # Sort regions by size in descending order
    props.sort(key=lambda x: x.area, reverse=True)
    
    # Create new binary image and fill largest region with white
    mask = np.zeros_like(image)
    mask[labels == props[0].label] = 255
    
    return mask

def ImageSeg(img):
    img = img
    args = get_args()
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    ## Image pre-processing section
    gray_image = img
    t = skimage.filters.threshold_otsu(gray_image)
    binary_mask = gray_image > t

    # Overlay the processed imgCopy with img
    img = skimage.img_as_float(gray_image)
    mask = binary_mask.astype(float)
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[:,:,0] = mask
    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    alpha = 0.7
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)

    # # Alternative way
    # # Use the binary mask to filter out pixels from the original image
    # # Pixels where the mask is False will be set to [0, 0, 0] (black)
    # img_masked = np.where(binary_mask[..., None], np.stack([img]*3, axis=-1), [0, 0, 0])

    img = Image.fromarray((img_masked * 255).astype(np.uint8))

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)

    result = mask_to_image(mask)
    footprint = disk(20);
    result = closing(result, footprint)
    mask = remove_small_objects(result)
    return mask