import sys
 
# setting path
sys.path.append('../TM2_segmentation')

import os
import SimpleITK as sitk
import numpy as np
import functools
from skimage.transform import resize
import nibabel as nib
import numpy as np
import pandas as pd

from scripts.preprocess_utils import load_nii, get_id_and_path

def slice_to_offset(input_annotation_file = 'data/all_metadata.csv', image_dir="data/z_scored_mris/z_with_pseudo/z/", 
targ_dir='data/slice_detection_w_pseudo2/',split_name='train'):
    # The z-offset for a slice should represent its offset above or below the L3 slice in mm 
    # Slices below the chosen L3 slice (closer to the feet) should be given negative offsets.
    print(split_name)

    csv_write_path = targ_dir + split_name + ".csv"
    df = pd.read_csv(input_annotation_file,header=0)

    df_to_be_writen = pd.DataFrame()
    df_train = df[df['train/test']==split_name]
    df_train=df_train[df_train['Ok registered? Y/N']=='Y'].reset_index()

    j = 0
    for idx in range(0, df_train.shape[0]):
        row = df_train.iloc[idx]
        patient_id, image_path, tm_file, _ = get_id_and_path(row, image_dir)
        print(patient_id, image_path, tm_file)

        if image_path != 0:
            seg_data, seg_affine = load_nii(tm_file)
            orientation = nib.aff2axcodes(seg_affine)
            
            # find the slice on which we made annotations
            if (np.asarray(nib.aff2axcodes(seg_affine))==['R', 'A', 'S']).all():
                slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
            
                image_sitk =  sitk.ReadImage(image_path)
                image_array  = sitk.GetArrayFromImage(image_sitk)
                spacing =  image_sitk.GetSpacing()[1]
                sp = image_sitk.GetSpacing()

                windowed_images = image_array
                if len(image_array)>3:
                    print(idx+1, patient_id, image_array.shape, np.max(image_array), spacing, slice_label)

                    resize_func = functools.partial(resize, output_shape=[256,256],
                                                preserve_range=True, anti_aliasing=True, mode='constant')
                    series = np.dstack([resize_func(im) for im in windowed_images])
                    series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])

                    for slice_idx in range(2,image_array.shape[0]-2):
                        offset = spacing*(slice_idx-slice_label)
                        offset = round(offset,3)
                        npy_name = str(j).zfill(6) + '_'+patient_id + '.npy'
                        npy_path = targ_dir + split_name + "/" + npy_name

                        image_array = np.zeros((256,256,1,5))
                        # create MIP from 5 neighboring slices to capture more information 
                        image_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.float32)
                        image_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.float32)
                        image_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.float32)
                        image_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.float32)
                        image_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.float32)
                        im_array= np.max(image_array, axis=3)

                        np.save(npy_path, im_array)

                        df_offset = pd.DataFrame({'NPY_name':npy_name,
                                    'ZOffset':offset,'slice_idx':slice_idx}, index=[0])
                        df_to_be_writen = pd.concat([df_to_be_writen, df_offset])
                        df_to_be_writen.to_csv(csv_write_path)   
                        j=j+1
                    print()
 
if __name__=="__main__":
    slice_to_offset(split_name='train')
    slice_to_offset(split_name='val')
