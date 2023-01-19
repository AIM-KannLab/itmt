import pandas as pd
import os
import numpy as np
import os
import shutil

AGE_FROM = 3
AGE_TO = 35+1
path = "/Volumes/kannlab_sfa$/Anna/TM_segmentation/data/pseudolabels_mris"

def find_file_in_path(name, path):
    result = []
    result = list(filter(lambda x:name in x, path))
    if len(result) != 0:
        return result[0]
    else:
        return False


# preprocessing ABCD-fmriresults01.txt dataset
df0 = pd.read_csv("/Volumes/kannlab_sfa$/Anna/TM_segmentation/data/csv_files/ABCD-fmriresults01.txt",delimiter="\t",header=0)
df0 = df0[["file_source", 'interview_age', "sex"]].dropna()
df0["sex"] = np.where(df0["sex"].str.contains("M"), 1, 2)
df0["SCAN_ID"] = df0["file_source"].str.split("/").str.get(-1).str.split("_ABCD").str.get(0)
df0["SCAN_PATH"] = "/Volumes/aertslab/USERS/Ben/BRAIN AGE/ABCD/Package_1200046/fmriresults01/abcd-mproc-release4"
#df0 = df0.drop(["file_source"], axis=1)

# select subset of those records that can be found in the folder
list_files = os.listdir("/Volumes/aertslab/USERS/Ben/BRAIN AGE/ABCD/Package_1200046/fmriresults01/abcd-mproc-release4")
df0["filename"] = df0["SCAN_ID"].apply(lambda x: find_file_in_path(x, list_files))
df0 = df0.loc[df0['filename'] != False]
print("df0 is done")

# preprocessing IXI.csv dataset
df1 = pd.read_csv("/Volumes/kannlab_sfa$/Anna/TM_segmentation/data/csv_files/IXI.csv",delimiter=",",header=0)
df1 = df1[["IXI_ID", 'AGE', "SEX_ID (1=m, 2=f)"]]
df1 = df1.rename({"IXI_ID": "subjectkey", "AGE": "interview_age", "SEX_ID (1=m, 2=f)": "sex"}).dropna()
df1['AGE'] = round(df1['AGE']*12, 0)
df1['SCAN_ID'] = "IXI" + df1["IXI_ID"].astype(str).str.zfill(3).astype(str)
df1['SCAN_PATH'] = "/Volumes/aertslab/USERS/Ben/BRAIN AGE/IXI_BrainDevelopment/IXI-T1w"

# select subset of those records that can be found in the folder
list_files = os.listdir("/Volumes/aertslab/USERS/Ben/BRAIN AGE/IXI_BrainDevelopment/IXI-T1w")
df1["filename"] = df1["SCAN_ID"].apply(lambda x: find_file_in_path(x, list_files))
df1 = df1.loc[df1['filename'] != False]
print("df1 is done")


# preprocessing PediatricMRI_DEMOGRAPHICS.csv dataset
df2 = pd.read_csv("/Volumes/kannlab_sfa$/Anna/TM_segmentation/data/csv_files/PediatricMRI_DEMOGRAPHICS.csv", delimiter=",", header=0)
df2 = df2[["PEDS_DEMOGRAPHICS01_ID", 'AGE_MONTHS_DOV_TO_DOB', "SUBJECT_GENDER", "SRC_SUBJECT_ID", "TIMEPOINT_LABEL"]]
df2 = df2.rename({"PEDS_DEMOGRAPHICS01_ID": "subjectkey", "AGE_MONTHS_DOV_TO_DOB": "interview_age"}).dropna()
df2["SUBJECT_GENDER"] = np.where(df2["SUBJECT_GENDER"].str.contains("Male"), 1, 2)
df2["SCAN_ID"] = "deface_" + df2['SRC_SUBJECT_ID'].astype(str) + "_" + df2["TIMEPOINT_LABEL"].astype(str).str.lower() + "_t1w.nii.gz"
df2["SCAN_PATH"] = "/Volumes/aertslab/USERS/Ben/BRAIN AGE/NDA-NIMH-PediatricMRI/Package_1200575/Package_1202659/peds_mri03/anatomicals/nifti/nonlongitudinally_registered/deface_t1w"
df2 = df2.drop(["SRC_SUBJECT_ID", "TIMEPOINT_LABEL"], axis=1)

# select subset of those records that can be found in the folder
list_files = os.listdir( "/Volumes/aertslab/USERS/Ben/BRAIN AGE/NDA-NIMH-PediatricMRI/Package_1200575/Package_1202659/peds_mri03/anatomicals/nifti/nonlongitudinally_registered/deface_t1w")
df2["filename"] = df2["SCAN_ID"].apply(lambda x: find_file_in_path(x, list_files))
#df2.to_csv(path_or_buf=path + "/df2.xls", header=True, index=False)
df2 = df2.loc[df2['filename'] != False]
print("df2 is done")

# convert to numpy for concatenation (to ignore the col names)
#np_df = df0.to_numpy()
#np_df1 = df1.to_numpy()
np_df2 = df2.to_numpy()

# concatenation
np_combined_data = np_df2#np.concatenate([np_df, np_df1, np_df2], axis=0)
df = pd.DataFrame(np_combined_data, columns=["ID", "AGE_M", "SEX", "SCAN_ID", "SCAN_PATH", "filename"])

# convert from months to years
df["AGE_M"] = (df["AGE_M"]/12).astype(int)

# shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


np_subsection = [["ID", "AGE_M", "SEX", "SCAN_ID", "SCAN_PATH", "filename"]]

for year in range(AGE_FROM, AGE_TO):
    subsection = df.loc[df['AGE_M'] == year][:4].to_numpy()
    print(subsection)
    np_subsection = np.concatenate([np_subsection, subsection], axis=0)
print(np.shape(np_subsection))

# write to the folder selected records and resulting .csv
for i in range(1, np.shape(np_subsection)[0]):
    shutil.copyfile(np_subsection[i, 4] + "/" + np_subsection[i, 5],
                    path + "/" + np_subsection[i, 5])

# write csv
df = pd.DataFrame(np_subsection)
df.to_csv(path_or_buf=path + "/Dataset_presudolabels_2.csv", index=False)

