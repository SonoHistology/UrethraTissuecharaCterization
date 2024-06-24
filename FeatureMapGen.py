# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:11:18 2024

@author: taiha
"""

import os
import logging
import SimpleITK as sitk
import six
from scipy.io import savemat

import radiomics
from radiomics import featureextractor, getFeatureClasses

def setup_progress_bar(package='tqdm'):
    """
    Setup progress bar for PyRadiomics computations using either 'tqdm' or 'click'.
    """
    if package == 'tqdm':
        import tqdm
        radiomics.progressReporter = tqdm.tqdm
    elif package == 'click':
        import click

        class progressWrapper:
            def __init__(self, iterable, desc=''):
                self.bar = click.progressbar(iterable, label=desc)

            def __iter__(self):
                return self.bar.__iter__()

            def __enter__(self):
                return self.bar.__enter__()

            def __exit__(self, exc_type, exc_value, tb):
                return self.bar.__exit__(exc_type, exc_value, tb)

        radiomics.progressReporter = progressWrapper

    radiomics.setVerbosity(logging.INFO)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='radiomics_log.txt', filemode='w',
                    format='%(levelname)s:%(name)s: %(message)s')

# Initialize feature extractor
paramsFile = os.path.abspath(r'Voxel.yaml')
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

# Setup progress bar (assumes tqdm or click is installed)
setup_progress_bar('tqdm')

# Define paths to the image and mask
imageName = 'img.tif'
maskName = 'seg.tif'

# Load the image and mask using SimpleITK
image = sitk.ReadImage(imageName)
mask = sitk.ReadImage(maskName)
featureVector = extractor.execute(imageName, maskName, voxelBased=True)

for featureName, featureValue in six.iteritems(featureVector):
    if isinstance(featureValue, sitk.Image):  # Check if featureValue is a SimpleITK Image
        # Convert to a NumPy array to store in a .mat file
        feature_array = sitk.GetArrayFromImage(featureValue)
        savemat(f'{featureName}.mat', {featureName: feature_array})
        print(f'Computed {featureName}, stored as "{featureName}.mat"')