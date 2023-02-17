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


split = "HEALTHY" # select the split to perform registration on
# this is created for some flexibility in reading metadata, and specifying input paths

if split == "TEST_CBN":
    input_path = "data/pre-processing/test"
    output_path = "data/registered/mni_templates/"
    metadata_file = "data/pre-processing/test/test_cbtn.csv"
elif split == "TEST_HBN":
    input_path = "data/pre-processing/test"
    output_path = "data/registered/mni_templates/"
    metadata_file = "data/pre-processing/test/test_hbn.xls"
elif split == "TRAIN":
    input_path = "data/pre-processing/train"
    output_path = "data/registered/mni_templates/"
    metadata_file = "data/pre-processing/train/train.xls"
elif split == "PSEUDO_LABELS":
    input_path = "data/pseudolabels_mris"
    output_path = "data/pseudolabels_registered/"
    metadata_file = "data/Dataset_presudolabels.csv"
elif split == "HEALTHY":
    input_path = "data/t1_mris/raw"
    output_path = "data/t1_mris/registered_not_ench/"
    metadata_file = "data/Dataset_t1_healthy_raw.csv"
    

# path_to_golden_template: {min_age, max_range}
age_ranges = {  "data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

# read in age marker and split into different age groups for template registration
if split == "TEST_CBN":
    df = pd.read_csv(metadata_file, header=0)
    for golden_file_path, age_values in age_ranges.items():
        df_t = df[df['AGE_M'].between(age_values["min_age"],age_values["max_age"])]
        for index, row in df_t.iterrows():
            filepath = find_file_in_path(row['SCAN_ID'], os.listdir(input_path))
            register_to_template(input_path+"/"+filepath, output_path, golden_file_path)

elif split == "TEST_HBN":
    df = pd.read_csv(metadata_file, sep="\t", header=0)
    for golden_file_path, age_values in age_ranges.items():
        df_t = df[df['Age'].between(age_values["min_age"],age_values["max_age"])]
        for index, row in df_t.iterrows():
            filepath = find_file_in_path(row['EID'], os.listdir(input_path))
            register_to_template(input_path+"/"+filepath, output_path, golden_file_path)

elif split == "PSEUDO_LABELS":
    df = pd.read_csv(metadata_file, sep=",", header=0)
    for golden_file_path, age_values in age_ranges.items():
        df_t = df[df['AGE_M'].between(age_values["min_age"],age_values["max_age"])]
        for index, row in df_t.iterrows():
            filepath = find_file_in_path(row['filename'], os.listdir(input_path))
            if len(filepath)>3:
                register_to_template(input_path+"/"+filepath, output_path, golden_file_path)
                
elif split == "HEALTHY":
    df = pd.read_csv(metadata_file, sep=",", header=0)
    df['AGE_Y'] = df['AGE_M'].astype('float32')//12
    for golden_file_path, age_values in age_ranges.items():
        df_t = df[df['AGE_Y'].between(age_values["min_age"],age_values["max_age"])]
        for index, row in df_t.iterrows():
            filepath = find_file_in_path(row['filename'], os.listdir(input_path))
            if len(filepath)>3:
                register_to_template(input_path+"/"+filepath, output_path, golden_file_path, create_subfolder=False)
    
elif split == "TRAIN":
    df = pd.read_csv(metadata_file, header=0)
    for golden_file_path, age_values in age_ranges.items():
        df_t = df[df['AGE_M'].between(age_values["min_age"],age_values["max_age"])]
        for index, row in df_t.iterrows():
            filepath = find_file_in_path(row['SCAN_ID'], os.listdir(input_path))
            register_to_template(input_path+"/"+filepath, output_path, golden_file_path)

            # extra preprocessing for extra formatting
            if "NDAR" in row['SCAN_ID']:
                file = tarfile.open(input_path+"/"+filepath)
                input_image_path=input_path+"/"+filepath
                new_path_extracted = "data/registered/NDAR/"+input_image_path.split("/")[-1].split(".")[0]
                
                file.extractall(new_path_extracted)
                file.close()
                stored_mri_path = new_path_extracted +"/sub-"+filepath.split("_")[0]+"/ses-"+filepath.split("_")[1]+"/anat"
                
                for file in os.listdir(stored_mri_path):
                    if file.endswith(".nii"):
                        print(file)
                        register_to_template(stored_mri_path+"/"+file, output_path, golden_file_path)

# next: registraion_mni.py
# after: preprocess_utils.py