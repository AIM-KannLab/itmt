import numpy as np
import pandas as pd
import nibabel as nib
import os
from skimage.restoration import inpaint
import scipy 
import skimage
from scipy.ndimage import binary_dilation

#todo:blurring should be done before the preprocesing step

path = 'data/bch_long_pre_test/raw/'
save_to = 'data/bch_long_pre_test/rand_noised/'
#for each path in path

def dilate_3d_binary_mask(mask, iterations=1):
    dilated_mask = binary_dilation(mask, iterations=iterations)
    return dilated_mask

for folder in os.listdir(path):
    if ".DS_Store" in folder or ".csv" in folder:
        continue
    for file in os.listdir(path + folder):
        if ".DS_Store" in file:
            continue
        if file.endswith('mask.nii.gz'):
            mask_path = path + folder + '/' + file
        else:
            name = file
            scan_path = path + folder + '/' + file
    mask = nib.load(mask_path).get_fdata().astype(np.bool)
    img = nib.load(scan_path).get_fdata()
    
    # create 3 options: 
    # blur the masked region with rangom noise
    min_pixel = np.min(img)
    max_pixel = np.max(img)
    
    # Step 1: Convert the mask to a binary mask (0s and 1s)
    mask_binary = mask.astype(np.uint8)

    # Step 2: Generate random noise of the same shape as the masked region
    #noise = np.random.uniform(min_pixel, max_pixel, size=img.shape).astype(np.float32)
    
    dilated_mask = dilate_3d_binary_mask(mask, 2)
    #subtract the mask from the dilated mask
    around_mask = np.logical_xor(dilated_mask, mask_binary)
    print(np.sum(around_mask), np.sum(mask_binary))
    
    pixel_values = img[around_mask]
    median_pixel_value = int(np.quantile(np.array(pixel_values),0.5))
    print(median_pixel_value)
    img_masked = np.where(mask_binary, median_pixel_value, img)
    #img_masked =  inpaint.inpaint_biharmonic(img, mask_binary, channel_axis=-1)
    
    '''median_pixel_value = int(np.quantile(np.array(img),0.75))
    # Step 3: Apply the noise to the masked region
    img_masked = np.where(mask_binary, median_pixel_value, img)
    '''
    # Step 4: Ensure the pixel values stay within the original range
    #img_masked = np.clip(img_masked, min_pixel, max_pixel)

    #save the image as a nifti file
    try:
        os.mkdir(save_to +'/' + name.split(".")[0])
    except:
        pass
    nib.save(nib.Nifti1Image(img_masked, affine=nib.load(scan_path).affine), save_to +'/' + name.split(".")[0] + '/' + name)
    #save the mask as a nifti file
    nib.save(nib.Nifti1Image(around_mask.astype(float), affine=nib.load(scan_path).affine), save_to +'/' + name.split(".")[0] + '/' + name.split(".")[0] + '_mask.nii.gz')
    #nib.save(nib.Nifti1Image(around_mask.astype(int), affine=nib.load(scan_path).affine), save_to +'/' + name.split(".")[0] + '/' + name.split(".")[0] + '_mask.nii.gz')
    
    