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

input_annotation_file = 'data/csv_files/calgary.csv'   
df = pd.read_csv(input_annotation_file, header=0)
input_path = "/mnt/kannlab_rfa/Anna/Calgary_preschool/T1_Dataset/"
save_to = 'data/t1_mris/calgary_reg/'

age_ranges = {  "data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

final_metadata = []

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
            
for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    age = row['Age (Years)'] 
    sex = row['Biological Sex (Female = 0; Male = 1)']
    if '0' in str(sex):
        sex = 2
    else:
        sex = 1
        
    for filepath in os.listdir(input_path):
        if str(row['PreschoolID']) in filepath:
            for golden_file_path, age_values in age_ranges.items():
                if age_values['min_age'] <= age and age <= age_values['max_age']: 
                    print(age, input_path+filepath+"/"+row['ScanID']+"/"+row['ScanID']+".nii.gz", save_to, golden_file_path)
                    register_to_template(input_path+filepath+"/"+row['ScanID']+"/"+row['ScanID']+".nii.gz", save_to, golden_file_path,row['ScanID'],create_subfolder=False)
                    #0,AGE_M,SEX,SCAN_PATH,Filename,dataset
                    final_metadata.append([int(age*12),sex,save_to+filepath+"/"+row['ScanID']+"/"+row['ScanID']+".nii.gz",row['ScanID']+".nii.gz",'Calgary'])
        #break
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_calgary.csv")

# Check cannot transform
# /mnt/kannlab_rfa/Anna/Calgary_preschool/T1_Dataset/10047/PS15_101/PS15_101.nii.gz data/t1_mris/calgary_reg/ data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii
#/mnt/kannlab_rfa/Anna/Calgary_preschool/T1_Dataset/10047/PS15_101/PS15_101.nii.gz

#3.9861 /mnt/kannlab_rfa/Anna/Calgary_preschool/T1_Dataset/10087/PS15_030/PS15_030.nii.gz data/t1_mris/calgary_reg/ data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii
#/mnt/kannlab_rfa/Anna/Calgary_preschool/T1_Dataset/10087/PS15_030/PS15_030.nii.gz
#Cannot transform PS15_030

# 3.9667 /mnt/kannlab_rfa/Anna/Calgary_preschool/T1_Dataset/10109/PS15_119/PS15_119.nii.gz data/t1_mris/calgary_reg/ data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii
#/mnt/kannlab_rfa/Anna/Calgary_preschool/T1_Dataset/10109/PS15_119/PS15_119.nii.gz
#Cannot transform PS15_119