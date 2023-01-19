import sys
# setting path
sys.path.append('../TM2_segmentation')

import os
import numpy as np
import nibabel as nib
import itk
import pandas as pd
import tarfile

from scripts.preprocess_utils import find_file_in_path

# load metadata file  

input_annotation_file = 'data/csv_files/NYU_2_phenotypic_data.csv'   
df = pd.read_csv(input_annotation_file, header=0)
input_path = "/media/sda/Anna/NUY_2/NYU_2/"
save_to = 'data/t1_mris/nyu_reg/'
df = df[df['SESSION']=="Baseline"]

age_ranges = {"data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

def register_to_template(input_image_path, output_path, fixed_image_path,rename_id,create_subfolder=True):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('data/golden_image/mni_templates/Parameters_Rigid.txt')

    if "nii" in input_image_path and "._" not in input_image_path:
        print(input_image_path)

        # Call registration function
        try:        
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            itk.imwrite(result_image, output_path+"/"+rename_id+".nii.gz")
                
            print("Registered ", rename_id)
        except:
            print("Cannot transform", rename_id)
            
final_metadata = []
#print(df)
for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    age = float(row['AGE_AT_SCAN_1'])  
    sex = row['SEX']
    print(age, sex)
    
    for filepath in os.listdir(input_path):
        if str(row['SUBID']).zfill(7) in filepath:
            for golden_file_path, age_values in age_ranges.items():
                if age_values['min_age'] <= float(age) and float(age) <= age_values['max_age']: 
                    path_container = input_path+filepath +"/session_1/anat_1/anat.nii.gz"
                    print(age, path_container, save_to, golden_file_path)
                    #0,AGE_M,SEX,SCAN_PATH,Filename,dataset
                    register_to_template(path_container, save_to, golden_file_path, filepath, create_subfolder=False)        
                    final_metadata.append([age*12,sex,path_container,filepath,'NYU'])
                    break
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_nyu.csv", header=['AGE_M','SEX','SCAN_PATH','Filename','dataset'])
# find . -name '.DS_Store' -type f -delete
