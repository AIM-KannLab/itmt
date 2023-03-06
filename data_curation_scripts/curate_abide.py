import sys
# setting path
sys.path.append('../TM2_segmentation')

import os
import numpy as np
import nibabel as nib
import itk
import pandas as pd
import tarfile

from scripts.preprocess_utils import find_file_in_path,register_to_template

# load metadata file  

input_annotation_file = 'data/csv_files/ABIDEII_Composite_Phenotypic.csv'   
df = pd.read_csv(input_annotation_file, header=0)
input_path = "/mnt/kannlab_rfa/Anna/ABIDE/zapaishchykova-20221211_194948/"
save_to = 'data/t1_mris/abide_registered/'

age_ranges = {"data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}
            
final_metadata = []
df = df[df['DX_GROUP']==2]
#print(df)
for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    age = row['AGE_AT_SCAN'] 
    sex = row['SEX']
    print(age, sex)
    for filepath in os.listdir(input_path):
        if str(row['SUB_ID']) in filepath:
            for subfolder in os.listdir(input_path+filepath):
                if "." not in subfolder:
                    nifti_path = input_path+filepath+"/"+subfolder+"/anat_1/NIfTI-1/anat.nii.gz"
                    for golden_file_path, age_values in age_ranges.items():
                        if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
                            print(age, nifti_path, save_to, golden_file_path)
                            register_to_template(nifti_path, save_to, golden_file_path, filepath, create_subfolder=False)
                            #0,AGE_M,SEX,SCAN_PATH,Filename,dataset
                            final_metadata.append([int(age*12),sex,save_to+filepath+".nii.gz",filepath+".nii.gz",'ABIDE'])
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_abide.csv")

