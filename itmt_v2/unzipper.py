import sys
import os
 
# setting path
sys.path.append('../TM2_segmentation')

import numpy as np
import pandas as pd
import nibabel as nib
from skimage.transform import resize, rescale
from scipy.ndimage import zoom
from settings import target_size_unet, scaling_factor

import matplotlib.pyplot as plt
from scripts.preprocess_utils import get_id_and_path,load_nii


# convert nifti to numpy array
def nifty_to_npy(df, path_images_array, path_masks_array):  
    # create empty arrays for populating with image slices
    images_array  = np.zeros((df.shape[0]*4, target_size_unet[0], target_size_unet[1], 1))
    masks_array  = np.zeros((df.shape[0]*4, target_size_unet[0], target_size_unet[1], 1))

    for idx, row in df.iterrows():
        print(idx)
        patient_id, image_path, tm_file, _ = get_id_and_path(row, image_dir)
        print(patient_id, image_path, tm_file)
        
        ## unzip image_path 
        