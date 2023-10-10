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

input_annotation_file = '/media/sda/Anna/brain_datasets/long579/ds003604-download/participants.tsv'   
df = pd.read_csv(input_annotation_file, header=0,delimiter='\t')
input_path = "/media/sda/Anna/brain_datasets/long579/ds003604-download/"
save_to = 'data/t1_mris/long579_reg/'

age_ranges = {"data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

final_metadata = []
#print(df)
#df = df.dropna(subset=['ses-5_date_ST', 'ses-7_date_ST','ses-9_date_ST'])
age_lst = [5,7,9]
already_processed = []
# leave onlyt those that have 3 scans
for idx in range(0,df.shape[0]):
    row = df.iloc[idx]
    sex = row['sex']
    if 'Female' in sex:
        sex = 2
    else:
        sex = 1
    print(str(row['participant_id']))
    for filepath in os.listdir(input_path):
        if str(row['participant_id']) in str(filepath) and filepath not in already_processed:
            print(filepath)
            for age in age_lst:  
                try:
                    for golden_file_path, age_values in age_ranges.items():
                        if age_values['min_age'] <= age and age <= age_values['max_age']: 
                            #/media/sda/Anna/long579/ds003604-download/sub-5003/ses-5/anat
                            print(age, input_path+filepath+"/ses-"+str(age)+"/anat/")
                            for mri_t1 in os.listdir(input_path+filepath+"/ses-"+str(age)+"/anat/"):
                                if "T1" in mri_t1 and ".nii" in mri_t1:
                                    register_to_template(input_path+filepath+"/ses-"+str(age)+"/anat/"+mri_t1, save_to, golden_file_path, create_subfolder=False)
                                    final_metadata.append([int(age*12),sex,save_to+"/"+filepath,input_path+filepath+"/ses-"+str(age)+"/anat/"+mri_t1,'LONG579'])
                except:
                    continue
            already_processed.append(filepath)
            
                            #0,AGE_M,SEX,SCAN_PATH,Filename,dataset
                        #final_metadata.append([int(age*12),sex,save_to+"/"+filepath,filepath+"_run-1_T1w.nii.gz",'AOMIC'])
            
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_long579.csv",header = ['AGE_M','SEX','SCAN_PATH','Filename','dataset'])
