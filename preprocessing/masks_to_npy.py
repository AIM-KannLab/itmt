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
from scripts.preprocess_utils import enhance, load_nii, get_id_and_path_not_nested,enhance_noN4,find_file_in_path

def get_id_and_path(row, image_dir):
    patient_id, image_path, tm_file = 0,0,0
    if row['Ok registered? Y/N'] == "N":
        print("skip - bad registration")
        return patient_id, image_path, tm_file
    if "NDAR" in row['Filename']:
        patient_id = row['Filename'].split("_")[0]
    else:
        patient_id = row['Filename'].split(".")[0]

    path = find_file_in_path(patient_id, os.listdir(image_dir))
    scan_folder = image_dir+path

    for file in os.listdir(scan_folder):
        if "._" in file: #skip hidden files
            continue
        t = image_dir+path+"/"+file
        if "TM" in file:
            tm_file = t
        if patient_id in file:
            image_path = t
    return patient_id, image_path, tm_file

def nifty_to_npy(df, path_images_array, path_masks_array):
    
    # create empty arrays for populating with image slices
    images_array  = np.zeros((df.shape[0]*4, target_size_unet[0], target_size_unet[1], 1))
    masks_array  = np.zeros((df.shape[0]*4, target_size_unet[0], target_size_unet[1], 1))

    for idx, row in df.iterrows():
        print(idx)
        patient_id, image_path, tm_file = get_id_and_path(row, image_dir)
        print(patient_id, image_path, tm_file)
        
        if image_path != 0:
            seg_data, seg_affine = load_nii(tm_file)
            if len(seg_data)>3:
                if (np.asarray(nib.aff2axcodes(seg_affine))==['R', 'A', 'S']).all():
                    slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 

                image_array, _ = load_nii(image_path)
                
                # rescale to 512x512
                rescaled_mask  = rescale(seg_data[:,15:-21,slice_label], scaling_factor).reshape(1,target_size_unet[0],target_size_unet[1],1) 
                rescaled_image = rescale(image_array[:,15:-21,slice_label], scaling_factor).reshape(1,target_size_unet[0],target_size_unet[1],1) 
                
                images_array[idx,:,:,:] = np.concatenate((rescaled_image[:,:256,:,:],np.zeros_like(rescaled_image[:,:256,:,:])),axis=1)
                masks_array[idx,:,:,:] = np.concatenate((rescaled_mask[:,:256,:,:],np.zeros_like(rescaled_mask[:,:256,:,:])),axis=1)
                images_array[df.shape[0]+idx,:,:,:] = np.concatenate((np.zeros_like(rescaled_image[:,256:,:,:]),rescaled_image[:,256:,:,:]),axis=1)
                masks_array[df.shape[0]+idx,:,:,:] = np.concatenate((np.zeros_like(rescaled_image[:,256:,:,:]),rescaled_mask[:,256:,:,:]),axis=1)
                images_array[2*df.shape[0]+idx,:,:,:] = np.concatenate((np.zeros_like(rescaled_image[:,:256,:,:]),rescaled_image[:,:256,:,:]),axis=1)
                masks_array[2*df.shape[0]+idx,:,:,:] = np.concatenate((np.zeros_like(rescaled_image[:,:256,:,:]),rescaled_mask[:,:256,:,:]),axis=1)
                images_array[3*df.shape[0]+idx,:,:,:] = np.concatenate((rescaled_image[:,256:,:,:],np.zeros_like(rescaled_image[:,256:,:,:])),axis=1)
                masks_array[3*df.shape[0]+idx,:,:,:] = np.concatenate((rescaled_mask[:,256:,:,:],np.zeros_like(rescaled_image[:,256:,:,:])),axis=1)
                
                print(idx, patient_id, np.shape(images_array[idx,:,:,:]),np.shape(masks_array[idx,:,:,:]))
    
    masks_array=masks_array.astype(np.uint8)
    np.save(path_images_array, images_array)
    np.save(path_masks_array, masks_array)

if __name__=="__main__":
    
    input_annotation_file = 'data/all_metadata.csv'
    image_dir  = 'data/z_scored_mris/z_with_pseudo/z/' #'data/denoised_mris/'

    df = pd.read_csv(input_annotation_file, header=0)
    df_train = df[df['train/test']=='train']
    df_train=df_train[df_train['Ok registered? Y/N']=='Y'].reset_index()
    df_val = df[df['train/test']=='val']
    df_val = df_val[df_val['Ok registered? Y/N']=='Y'].reset_index()
    
    nifty_to_npy(df_train, path_images_array='data/z_segmentations_pseudo/train_images.npy', path_masks_array='data/z_segmentations_pseudo/train_masks.npy')
    nifty_to_npy(df_val, path_images_array='data/z_segmentations_pseudo/val_images.npy', path_masks_array='data/z_segmentations_pseudo/val_masks.npy')



