import sys
sys.path.insert(1, '../')
# setting path
sys.path.append('../TM2_segmentation')

import numpy as np
import pandas as pd
import os
import shutil
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


input_annotation_file = 'itmt_v2.0/itmt2.0.csv'
df = pd.read_csv(input_annotation_file, header=0)

image_dir = '/media/sda/Anna/TM2_segmentation/data/itmt2.0/'

not_found = []
for idx, row in df.iterrows():
    #print(idx)
    patient_id, image_path, tm_file, _ = get_id_and_path(row, image_dir)
    if  os.path.exists(image_path)==False or os.path.exists(tm_file)==False:
        print(patient_id,df['Dataset'].iloc[idx], image_path, tm_file)
        not_found.append([patient_id,df['Dataset'].iloc[idx],df['AGE_M'].iloc[idx],df['SEX'].iloc[idx]])
        
pd.DataFrame(not_found, columns=['ID','Dataset','AGE_M','SEX']).to_csv('/media/sda/Anna/TM2_segmentation/itmt_v2.0/not_found.csv', index=False)
        
        
