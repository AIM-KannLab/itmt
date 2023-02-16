import sys
# setting path
sys.path.append('../TM2_segmentation')

import os
import numpy as np
import nibabel as nib
import itk
import pandas as pd
import tarfile
from dateutil import parser
from datetime import datetime
from scripts.preprocess_utils import find_file_in_path, register_to_template

# load metadata file  
def isNaN(string):
    return string != string

input_annotation_file = 'data/DMG_MRI_nifti/converted_dicoms_metadata.csv'   
df = pd.read_csv(input_annotation_file, header=0)
input_path = "data/DMG_MRI_nifti/"
save_to = 'data/t1_mris/dmg_reg/'

age_ranges = {"data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

final_metadata = []
#print(df)
for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    age = int(row['years_at_scan'])
    
    sex = row['sex']
    if 'F' in sex:
        sex = 2
    else:
        sex = 1

    for filepath in os.listdir(input_path):
        #print(filepath,str(row['BCH MRN']).split(".")[0].zfill(7))
        if str(row['id']) in filepath:
            for golden_file_path, age_values in age_ranges.items():
                if age_values['min_age'] <= age and age <= age_values['max_age']: 
                    print(age, input_path+filepath, save_to, golden_file_path)
                    register_to_template(input_path+filepath, save_to, golden_file_path, create_subfolder=False)
                    #0,AGE_M,SEX,SCAN_PATH,Filename,dataset
                    final_metadata.append([age,sex,save_to+filepath,filepath,'DMG'])
                    break
       # break
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_dmg.csv",header = ['AGE_M','SEX','SCAN_PATH','Filename','dataset'])

