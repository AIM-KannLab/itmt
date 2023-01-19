import pandas as pd
import os
import numpy as np
import os
import shutil
import subprocess
import re

AGE_FROM = 3 * 12
AGE_TO = 35 * 12

path_to_save = "data/t1_mris/raw/"

def find_file_in_path(name, path):
    result = []
    result = list(filter(lambda x:name in x, path))
    if len(result) != 0:
        return result[0]
    else:
        return False

def find_file_in_path_HBN_S3(name, path):
    result = False
    for root, dirs, files in os.walk(path):
        result = next((os.path.join(file) for file in files if name in file), False)
    return result

# preprocessing ABCD-fmriresults01.txt dataset
df0 = pd.read_csv("data/csv_files/ABCD-fmriresults01.txt",delimiter="\t",header=0)
df0 = df0[["file_source", 'interview_age', "sex"]].dropna()
df0["sex"] = np.where(df0["sex"].str.contains("M"), 1, 2)
df0 = df0.rename({"interview_age": "AGE_M", "sex": "SEX"}, axis=1).dropna()

#from NDARINV0314RN9P_ses-2YearFollowUpYArm1 into sub-NDARINVG0F3TJPW_ses-2YearFollowUpYArm1_run-01_T2w.nii
df0["SCAN_ID"] = df0["file_source"].str.split("/").str.get(-1).str.split("_ABCD").str.get(0).str.split("_").str.get(0) + "_ses-" +\
    df0["file_source"].str.split("/").str.get(-1).str.split("_ABCD").str.get(0).str.split("_").str.get(1)
df0["SCAN_PATH"] = "/mnt/aertslab/USERS/Ben/BRAIN AGE/ABCD/Package_1200046/extracted_tgz_T1_anna/"

# select subset of those records that can be found in the folder
#/mnt/aertslab/USERS/Ben/BRAIN AGE/ABCD/Package_1200050/extracted_tgz_T2_anna/
list_files = os.listdir("/mnt/aertslab/USERS/Ben/BRAIN AGE/ABCD/Package_1200046/extracted_tgz_T1_anna")
df0["filename"] = df0["SCAN_ID"].apply(lambda x: find_file_in_path(x, list_files))
df0 = df0.loc[df0['filename'] != False]
df0['dataset']='ABCD'
df0=df0[["AGE_M", "SEX", "SCAN_PATH", "filename","dataset"]]
print("df0 is done")

#____________________________________
# preprocessing IXI.csv dataset
df1 = pd.read_csv("data/csv_files/IXI.csv",delimiter=",",header=0)
df1 = df1[["IXI_ID", 'AGE', "SEX_ID (1=m, 2=f)"]]
df1 = df1.rename({"AGE": "AGE_M", "SEX_ID (1=m, 2=f)": "SEX"}, axis=1).dropna()
df1['AGE_M'] = (df1['AGE_M']*12).astype(int)
df1['SCAN_ID'] = "IXI" + df1["IXI_ID"].astype(str).str.zfill(3).astype(str)
df1['SCAN_PATH'] = "/mnt/aertslab/USERS/Ben/BRAIN AGE/IXI_BrainDevelopment/IXI-T1w"

# select subset of those records that can be found in the folder
list_files = os.listdir("/mnt/aertslab/USERS/Ben/BRAIN AGE/IXI_BrainDevelopment/IXI-T1w")
df1["filename"] = df1["SCAN_ID"].apply(lambda x: find_file_in_path(x, list_files))
df1 = df1.loc[df1['filename'] != False]
df1['dataset']='IXI'

df1=df1[["AGE_M", "SEX", "SCAN_PATH", "filename",'dataset']]
print("df1 is done")

#____________________________________
# preprocessing PediatricMRI_DEMOGRAPHICS.csv dataset
df2 = pd.read_csv("data/csv_files/PediatricMRI_DEMOGRAPHICS.csv", delimiter=",", header=0)
df2 = df2[["PEDS_DEMOGRAPHICS01_ID", 'AGE_MONTHS_DOV_TO_DOB', "SUBJECT_GENDER", "SRC_SUBJECT_ID", "TIMEPOINT_LABEL"]]

df2["SEX"] = np.where(df2["SUBJECT_GENDER"].str.contains("Male"), 1, 2)
df2["SCAN_ID"] = "clamp_" + df2['SRC_SUBJECT_ID'].astype(str) + "_" + df2["TIMEPOINT_LABEL"].astype(str).str.lower() + "_t2w.nii.gz"
df2["SCAN_PATH"] = "/mnt/aertslab/USERS/Ben/BRAIN AGE/NDA-NIMH-PediatricMRI/Package_1200575/Package_1202659/peds_mri03/anatomicals/nifti/nonlongitudinally_registered/deface_t1w"
#df2 = df2.drop(["SRC_SUBJECT_ID", "TIMEPOINT_LABEL"], axis=1)
df2 = df2.rename({"AGE_MONTHS_DOV_TO_DOB": "AGE_M"}, axis=1).dropna()

# select subset of those records that can be found in the folder
list_files = os.listdir("/mnt/aertslab/USERS/Ben/BRAIN AGE/NDA-NIMH-PediatricMRI/Package_1200575/Package_1202659/peds_mri03/anatomicals/nifti/nonlongitudinally_registered/deface_t1w")
df2["filename"] = df2["SCAN_ID"].apply(lambda x: find_file_in_path(x, list_files))
#df2.to_csv(path_or_buf=path + "/df2.xls", header=True, index=False)
df2 = df2.loc[df2['filename'] != False]

df2['dataset']='NIMH'
df2=df2[["AGE_M", "SEX", "SCAN_PATH", "filename",'dataset']]
print("df2 is done")

# preprocessing baby connectome dataset
df3= pd.read_csv("/mnt/kannlab_rfa/Anna/data_baby_connectome/image03/botht1andt2.csv",delimiter=",",header=0)
df3 = df3[["IMAGE_FILE", 'INTERVIEW_AGE', "SEX"]].dropna()
df3["SEX"] = np.where(df3["SEX"].str.contains("M"), 1, 2)
df3 = df3.rename({"INTERVIEW_AGE": "AGE_M"}, axis=1).dropna()

#from s3://NDAR_Central_2/submission_31933/BCP/xnat_archive_BCP_arc001_subjects/MNBCP116056/v02-1-6mo-20170324/NRRD/20170324-ST001-Elison_BSLERP_116056_02_01_MR-SE003-T1w.nii.gz
df3["SCAN_ID"] = df3["IMAGE_FILE"].str.split("/").str.get(-1)
df3["SCAN_PATH"] = "/mnt/kannlab_rfa/Anna/data_baby_connectome/image03/BCP/xnat_archive_BCP_arc001_subjects/"+ df3["IMAGE_FILE"].str.split("xnat_archive_BCP_arc001_subjects/").str.get(-1).str.split("NRRD/").str.get(0)+"NRRD/"

# select subset of those records that can be found in the folder
filenames_numpy = []
for index, row in df3.iterrows():
    list_files = os.listdir(row['SCAN_PATH'])
    if "T1w" in row['SCAN_ID']:
        filenames_numpy.append(find_file_in_path(row['SCAN_ID'],list_files))
    else:
        filenames_numpy.append(False)

df3['filename'] = filenames_numpy
df3 = df3.loc[df3['filename'] != False]
df3['dataset']='BABY'
df3=df3[["AGE_M", "SEX", "SCAN_PATH", "filename", "dataset"]]
print("df3 is done")

#____________________________________
# convert to numpy for concatenation (to ignore the col names)
np_df = df0.to_numpy()
np_df1 = df1.to_numpy()
np_df2 = df2.to_numpy()
np_df3 = df3.to_numpy()

# concatenation
np_combined_data = np.concatenate([np_df, np_df1, np_df2, np_df3], axis=0)
df = pd.DataFrame(np_combined_data, columns=["AGE_M", "SEX", "SCAN_PATH", "filename",'dataset'])

np_subsection = [["AGE_M", "SEX", "SCAN_PATH", "filename",'dataset']]

for month in range(AGE_FROM, AGE_TO):
    subsection = df.loc[df['AGE_M'] == month].to_numpy()
    np_subsection = np.concatenate([np_subsection, subsection], axis=0)
print(np.shape(np_subsection))

df = pd.DataFrame(np_subsection, columns=["AGE_M", "SEX", "SCAN_PATH", "filename", 'dataset'])
df.to_csv(path_or_buf= "data/Dataset_t1_healthy_raw.csv",header =["AGE_M", "SEX", "SCAN_PATH", "filename", 'dataset'], index=True)

# write to the folder selected records and resulting .csv
for i in range(1, np.shape(np_subsection)[0]):
    print(i)
    try:
        shutil.copyfile(np_subsection[i, 2] + "/" + np_subsection[i, 3],
                 path_to_save + np_subsection[i, 3])
    except:
        continue

# next: registraion_mni.py
# after: preprocess_utils.py
