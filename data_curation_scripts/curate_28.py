import sys
# setting path
sys.path.append('../TM2_segmentation')

import os
import numpy as np
import nibabel as nib
import itk
import pandas as pd
import tarfile

from scripts.preprocess_utils import find_file_in_path, register_to_template

# load metadata file  

input_annotation_file = '/media/sda/Anna/28andme/ds002674-download/participants.tsv'   
df = pd.read_csv(input_annotation_file, header=0,delimiter='\t')
input_path = "/media/sda/Anna/28andme/ds002674-download/sub-01/"
save_to = 'data/t1_mris/28_reg/'
#clean input path

age_ranges = {  "data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

final_metadata = []
#this is one subject study
for idx in range(0, 1):
    row = df.iloc[idx]
    age = 23
    sex = 2
    
    for filepath in os.listdir(input_path):
        golden_file_path = "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii"
        print(input_path+filepath)
        for img in os.listdir(input_path+filepath+"/anat"):
            if "T1w" in img and "nii" in img:
                print(age, input_path+filepath+"/anat/"+img, save_to, golden_file_path)
                register_to_template(input_path+filepath+"/anat/"+img, save_to, golden_file_path, create_subfolder=False)
                final_metadata.append([int(age*12),sex,input_path+filepath+"/anat/"+img,img,'28'])
                
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_28.csv", header=['AGE_M','SEX','SCAN_PATH','Filename','dataset'])

