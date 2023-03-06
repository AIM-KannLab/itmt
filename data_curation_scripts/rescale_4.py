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

input_annotation_file = '/media/sda/Anna/TM2_segmentation/data/dices_healthy.csv'
df = pd.read_csv(input_annotation_file, header=0)
input_path = "/media/sda/Anna/TM2_segmentation/data/rescaling_experiments/raw_down4/"
save_to = '/media/sda/Anna/TM2_segmentation/data/rescaling_experiments/raw_down4_reg/'

age_ranges = {"data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

final_metadata = []
for idx in range(0,df.shape[0]):
    row = df.iloc[idx]
    sex = row['Gender']
    age = row['Age']
    for filepath in os.listdir(input_path):
        if str(row['ID']) in str(filepath):
            try:
                for golden_file_path, age_values in age_ranges.items():
                    if age_values['min_age'] <= age and age <= age_values['max_age']: 
                        print(age, input_path+filepath)
                        register_to_template(input_path+filepath, save_to, golden_file_path, create_subfolder=False)
                        final_metadata.append([int(age*12),sex,save_to+"/"+filepath,input_path+filepath,'Down2'])
            except:
                continue
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_down4.csv",header = ['AGE_M','SEX','SCAN_PATH','Filename','dataset'])

