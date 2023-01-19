import sys
sys.path.append('../TM2_segmentation')

import pandas as pd
import os
import numpy as np
import os
import shutil
import subprocess
import re
from settings import AGE_FROM, AGE_TO

path = "/media/sda/Anna/TM2_segmentation/data/curated_test/"

def find_file_in_path(name, path):
    result = []
    result = list(filter(lambda x:name in x, path))
    if len(result) != 0:
        return result[0]
    else:
        return False

# preprocessing baby connectome dataset
df6= pd.read_csv("/mnt/kannlab_rfa/Anna/data_baby_connectome/image03/botht1andt2.csv",delimiter=",",header=0)
df6 = df6[["IMAGE_FILE", 'INTERVIEW_AGE', "SEX"]].dropna()
df6["SEX"] = np.where(df6["SEX"].str.contains("M"), 1, 2)
df6 = df6.rename({"INTERVIEW_AGE": "AGE_M"}, axis=1).dropna()

#from s3://NDAR_Central_2/submission_31933/BCP/xnat_archive_BCP_arc001_subjects/MNBCP116056/v02-1-6mo-20170324/NRRD/20170324-ST001-Elison_BSLERP_116056_02_01_MR-SE003-T1w.nii.gz
df6["SCAN_ID"] = df6["IMAGE_FILE"].str.split("/").str.get(-1)
df6["SCAN_PATH"] = "/mnt/kannlab_rfa/Anna/data_baby_connectome/image03/BCP/xnat_archive_BCP_arc001_subjects/"+ df6["IMAGE_FILE"].str.split("xnat_archive_BCP_arc001_subjects/").str.get(-1).str.split("NRRD/").str.get(0)+"NRRD/"

# select subset of those records that can be found in the folder
filenames_numpy = []
for index, row in df6.iterrows():
    list_files = os.listdir(row['SCAN_PATH'])
    if "T1w" in row['SCAN_ID']:
        filenames_numpy.append(find_file_in_path(row['SCAN_ID'],list_files))
    else:
        filenames_numpy.append(False)

df6['filename'] = filenames_numpy
df6 = df6.loc[df6['filename'] != False]
df6=df6[["AGE_M", "SEX", "SCAN_PATH", "filename"]]
df6['AGE_Y'] = df6['AGE_M']//12
print("curation is done")

df6.to_csv(path_or_buf=path+"test_baby.csv", index=True)

np_subsection = [["AGE_M", "SEX", "SCAN_PATH", "filename","AGE_Y"]]
for year in range(AGE_FROM, AGE_TO):
    subsection = df6.loc[df6['AGE_Y'] == year][:3].to_numpy()
    np_subsection = np.concatenate([np_subsection, subsection], axis=0)
    
print(np_subsection)
print(np.shape(np_subsection))

for i in range(1, np.shape(np_subsection)[0]):
    print(i)
    shutil.copyfile(np_subsection[i, 2] + "/" + np_subsection[i, 3],\
        path + "raw/" + np_subsection[i, 3])


df = pd.DataFrame(np_subsection)
df.to_csv(path_or_buf="data/curated_test/test_baby.csv", index=False)