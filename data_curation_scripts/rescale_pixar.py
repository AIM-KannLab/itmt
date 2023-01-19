import skimage.transform as skTrans
import nibabel as nib
import numpy as np
from nibabel.affines import rescale_affine
import os
import pandas as pd
import logging
import SimpleITK as sitk
from scipy.signal import medfilt
import itk
import skimage
import functools
from skimage.transform import resize
import subprocess
import shutil
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.zscore import ZScoreNormalize

def crop_center(img, cropx,cropy,cropz):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)  
    startz = z//2-(cropz//2)    
    return img[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]

def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return

def find_in_metafile(file, df):
    for index, row in df.iterrows():
        if row['participant_id'] in file:
            return row['Age'], row['Gender'],0#row['dataset']
    print("not found", file)
    not_found.append(file)
    return 0,0,0

def find_file_in_path(name, path):
    result = []
    result = list(filter(lambda x:name in x, path))
    if len(result) != 0:
        return result[0]
    else:
        return False

def register_to_template(input_image_path, fixed_image_path):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('data/golden_image/mni_templates/Parameters_Rigid.txt')

    if ".nii" in input_image_path:
        print(input_image_path)
        # Call registration function
        try:
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            #itk.imwrite(result_image, output_path+"/"+image_id)
            print("Registered ", image_id)
            return 0,result_image
        except:
            print("Cannot transform", input_image_path.split("/")[-1])
            return 1,1
   
def select_template_based_on_age(age):
    #https://nist.mni.mcgill.ca/atlases/
    if age>=3 and age<= 7:
        return 'data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii'
    if age> 7 and age<=13:
        return 'data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii'
    if age> 13 and age<=35:
        return 'data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii'
    return 0

def load_nii(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return

def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)
    
def apply_window(image, win_centre= 40, win_width= 400):
    range_bottom = 149 #win_centre - win_width / 2
    scale = 256 / 256 #win_width
    image = image - range_bottom

    image = image * scale
    image[image < 0] = 0
    image[image > 255] = 255

    return image

def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    #remove background pixels by the otsu filtering
    t = skimage.filters.threshold_otsu(volume,nbins=6)
    volume[volume < t] = 0
    
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])
    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

def enhance_noN4(volume, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    try:
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        return volume
    except RuntimeError:
        logging.warning('Failed enchancing')

if __name__=="__main__":
    raw_path = 'data/curated_test/pixar_raw/'
    metadata_file = "data/curated_test/participants.tsv"
    df = pd.read_csv(metadata_file,delimiter="\t",header=0)
    np_subsection = []
    not_found = []

    output_path = "data/curated_test/registered_not_ench/"

    for i in range(0,len(os.listdir(raw_path))):
        print(i, os.listdir(raw_path)[i])
        file = os.listdir(raw_path)[i]
        file_path = raw_path + file
        if ".tgz" in file_path or "._" in file_path:
            os.remove(file_path)
        if ".nii" in file_path and "._" not in file_path:
            im = nib.load(file_path).get_fdata()
            age, sex, dataset_label = find_in_metafile(file, df)
            if sex != 0:
                age_y = int(age)
                golden_file_path = select_template_based_on_age(age_y)
               
                error_code, registered = register_to_template(file_path, golden_file_path)
                
                if golden_file_path != 0:
                    #image_array = enhance_noN4(itk.GetArrayFromImage(registered))
                    
                    # perform z-norm
                    #z_norm = ZScoreNormalize()
                    #result = z_norm(image_array, modality=Modality.T1)
                    image3 = sitk.GetImageFromArray(registered)
                    res_path = output_path+file.split("/")[-1].split(".")[0]
                    os.mkdir(res_path)
                    print(res_path+file)
                    sitk.WriteImage(image3, res_path+"/"+file) 
                    if sex == "F":
                        sex_int = 2
                    else: 
                        sex_int = 1
                    np_subsection.append([file_path,age_y,sex_int,dataset_label])
                    #print(np_subsection)
                    #break
                    
    #  write those that were preprocesses                           
    df = pd.DataFrame(np_subsection)
    df.to_csv(path_or_buf="data/curated_test/Dataset_test_rescaled_pixar.csv", header=False, index=False)

    # for the creating t1 dataset for the TM measurements: 
    # run curate_healthy.py
    # run curate_s3_dataset.py
    # merge s3 and dataset
    # run rescale_healthy.py
    # do the inference