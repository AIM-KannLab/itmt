import sys
sys.path.insert(1, '../')

import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime

metadata_dob = pd.read_csv('itmt_v2/cvs_files/metadata_bch_scan_date.csv', header=0)
def find_age_gender(id_to_find):  
    for idx, row in metadata_dob.iterrows():
        if str(id_to_find).zfill(7)==str(row['BCH_MRN']).zfill(7):
            #12/20/93
            dob=datetime.strptime(row['dob'], "%m/%d/%y")
            sex=row['sex']
            mri_date=datetime.strptime(row['scan_date'], "%m/%d/%y")
    age = (mri_date-dob).days/365
    if sex == "F":
        sex=2
    else:  
        sex=1
    return round(age,2), sex
            
        
path = 'data/itmt2.0_checkedBK_bch/'
path_to = 'data/itmt2.0/'
df = pd.read_csv('data/itmt2.0_checkedBK_bch/annotations.csv', header=None)
df= df[df[1]==1] # select only the good cases

save_metadata = []

for idx in range(0,len(df)):
    file_name = df.iloc[idx][0]
    new_path = path_to+file_name.split(".")[0]+"/"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        shutil.copy(path+file_name.split(".")[0]+"_mask.nii.gz", new_path+"TM.nii.gz")
    
    shutil.copy('data/t1_mris/bch_reg_ench/z/'+file_name.split(".")[0]+'/'+file_name.split(".")[0]+'.nii', new_path+file_name.split(".")[0]+'.nii.gz')
    age, sex = find_age_gender(file_name.split(".")[0])
    save_metadata.append([file_name,age, sex,'BCH'])
    print(file_name,age, sex)
    
pd.DataFrame(save_metadata).to_csv("itmt_v2/cvs_files/itmt2.0_cancer.csv", header=["Filename","AGE_M","SEX","Dataset"], index=None)
#Filename,AGE_M,SEX,Dataset,train/test,Ok registered? Y/N