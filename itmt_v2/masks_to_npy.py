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
from scripts.preprocess_utils import load_nii,find_file_in_path


# helper function to get the id and path of the image and the mask
def get_id_and_path(row, image_dir, nested = False, no_tms=True):
    patient_id, image_path, ltm_file, rtm_file = "","","",""
    patient_id = str(row['Filename']).split(".")[0].split("/")[-1]
    path = find_file_in_path(patient_id, os.listdir(image_dir))

    scan_folder = image_dir+path
    
    for file in os.listdir(scan_folder):
        t = image_dir+path+"/"+file
        if "LTM" in file:
            ltm_file = t
        elif "RTM" in file:
            rtm_file = t
        elif "TM" in file:
            rtm_file = t
            ltm_file = t
        if patient_id in file:
            image_path = t
    return patient_id, image_path, ltm_file, rtm_file

# convert nifti to numpy array
def nifty_to_npy(df, path_images_array, path_masks_array):  
    # create empty arrays for populating with image slices
    images_array  = np.zeros((df.shape[0]*4, target_size_unet[0], target_size_unet[1], 1))
    masks_array  = np.zeros((df.shape[0]*4, target_size_unet[0], target_size_unet[1], 1))

    for idx, row in df.iterrows():
        #print(idx)
        patient_id, image_path, tm_file, _ = get_id_and_path(row, image_dir)
        #NDAR processed differently
        
        if image_path != 0:
            seg = nib.load(tm_file)
            seg_data, seg_affine  = seg.get_fdata(), seg.affine
            if len(seg_data)>3:
                #print(np.asarray(nib.aff2axcodes(seg_affine)))
                     
                if 'gz' in image_path :
                    os.rename(image_path,image_path.split('.gz')[0])
                    image_path = image_path.split('.gz')[0]
                    image_array, img_aff = load_nii(image_path)
                else:
                    image_array, img_aff = load_nii(image_path)
                
                if (np.asarray(nib.aff2axcodes(seg_affine))==['L', 'P', 'S']).all() or\
                    (np.asarray(nib.aff2axcodes(seg_affine))==['R', 'A', 'S']).all():
                    slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
                else:
                    print("Wrong orientation cannot infer slice label")
               
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
                
                #print(idx, patient_id, np.shape(images_array[idx,:,:,:]),np.shape(masks_array[idx,:,:,:]))
                if np.sum(masks_array[idx,:,:,:])<=15 or np.sum(masks_array[df.shape[0]+idx,:,:,:])<=15\
                    or np.sum(masks_array[2*df.shape[0]+idx,:,:,:])<=15 or np.sum(masks_array[3*df.shape[0]+idx,:,:,:])<=15:
                    print("smal", patient_id, image_path, tm_file,np.asarray(nib.aff2axcodes(seg_affine)))
        
                #break
            
    masks_array=masks_array.astype(np.uint8)
    np.save(path_images_array, images_array)
    np.save(path_masks_array, masks_array)

if __name__=="__main__": 
    input_annotation_file = 'itmt_v2/itmt2.0_joint.csv'
    image_dir  = 'data/itmt2.0/'

    df = pd.read_csv(input_annotation_file, header=0)
    df_train = df[df['train/test']=='train'].reset_index()
    #df_train=df_train[df_train['Ok registered? Y/N']=='Y'].reset_index()
    df_val = df[df['train/test']=='val'].reset_index()
    #df_val = df_val[df_val['Ok registered? Y/N']=='Y'].reset_index()
    
    nifty_to_npy(df_train, path_images_array='data/segmentations_itmt2.0/train_images.npy', 
                 path_masks_array='data/segmentations_itmt2.0/train_masks.npy')
    nifty_to_npy(df_val, path_images_array='data/segmentations_itmt2.0/val_images.npy', 
                 path_masks_array='data/segmentations_itmt2.0/val_masks.npy')



