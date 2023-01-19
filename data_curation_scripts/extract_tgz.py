import os
import numpy as np
import nibabel as nib
import pandas as pd
import tarfile
import shutil

#extracted up to 8841
path_from = '/mnt/aertslab/USERS/Ben/BRAIN AGE/ABCD/Package_1200046/fmriresults01/abcd-mproc-release4/' #"/Volumes/aertslab/USERS/Anna/ModelsGenesis/keras/data/ngz/"
extract_to_path = '/mnt/aertslab/USERS/Ben/BRAIN AGE/ABCD/Package_1200046/extracted_tgz_T1_anna/'
print(len(os.listdir(path_from)))
for i in range(8841,len(os.listdir(path_from))):
    id_patient = os.listdir(path_from)[i]
    print(i, id_patient)
    if ".tgz" in id_patient:
        filepath = path_from + id_patient
        file = tarfile.open(filepath)

        #  extracting file
        temp_folder = extract_to_path+"c/"
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
        try:
            file.extractall(temp_folder)
            file.close()

            extrated_filepath = temp_folder + "sub-" + id_patient.split("_")[0] + "/ses-" + id_patient.split("_")[1] + "/anat/"
            print(extrated_filepath)
            
            for file_nii in os.listdir(extrated_filepath):
                if ".nii" in file_nii:
                    #copy from sub-NDARINV03KMHMJJ/ses-baselineYear1Arm1/anat to root
                    shutil.copyfile(extrated_filepath + file_nii, extract_to_path + file_nii) #path + file_nii)
            
            # delete path and file
            shutil.rmtree(temp_folder, ignore_errors=False, onerror=None)
        except:
            continue
