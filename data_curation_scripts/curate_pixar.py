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

input_annotation_file = 'data/csv_files/pixar.tsv'   
df = pd.read_csv(input_annotation_file, header=0,delimiter='\t')
input_path = "/mnt/kannlab_rfa/Anna/Pixar/ds000228_R1.0.1/"
save_to = 'data/t1_mris/pixar/'

age_ranges = {  "data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

final_metadata = []

for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    age = row['Age'] 
    sex = row['Gender']
    for filepath in os.listdir(input_path):
        if row['participant_id'] in filepath:
            for golden_file_path, age_values in age_ranges.items():
                if age_values['min_age'] <= age and age <= age_values['max_age']: 
                    print(age, input_path+filepath+"/anat/"+filepath+"_T1w.nii.gz", save_to, golden_file_path)
                    #register_to_template(input_path+filepath+"/anat/"+filepath+"_T1w.nii.gz", save_to, golden_file_path, create_subfolder=False)
                    #0,AGE_M,SEX,SCAN_PATH,Filename,dataset
                    final_metadata.append([int(age*12),sex,save_to+"/"+filepath,filepath+"_T1w.nii.gz",'Pixar'])

df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_pixar.csv")

