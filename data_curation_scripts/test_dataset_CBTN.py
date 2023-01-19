import os
import pandas as pd
import numpy as np
import shutil
import subprocess
from settings import AGE_FROM, AGE_TO

N_CASES = 2
path_for_images = "./data/pre-processing/test"
storage_path = "Z:/USERS/Zezhong/pLGG/CBTN_BCH_Data/T1W"

def find_file_in_path(name, path):
    result = []
    result = list(filter(lambda x:name in x, path))
    if len(result) != 0:
        return result[0]
    else:
        return False

# read in csv
df = pd.read_csv("./data/cbtn-all-LGG-BRAF.csv", delimiter="," , header=0)
df=df[["CBTN Subject ID", "Age at Diagnosis", "Gender", "Diagnosis Type"]]
df = df[df["Age at Diagnosis"]!="Not Reported"]
df = df[df["Diagnosis Type"]=="Initial CNS Tumor"] # get only initial CNS cases for age
print(df)
df["Age at Diagnosis"]=df['Age at Diagnosis'].astype(int).div(356).round(0)
df['Occur'] = df.groupby('CBTN Subject ID')['CBTN Subject ID'].transform('size')
# remove the multiple occurence subjects from the selection
# print(df[df["Occur"]!=1])
df = df[df["Occur"]==1]
df=df[["CBTN Subject ID", "Age at Diagnosis", "Gender"]]

list_files = os.listdir(storage_path)
df["filename"] = df["CBTN Subject ID"].apply(lambda x: find_file_in_path(x, list_files))
df["Gender"] = np.where(df["Gender"].str.contains("Male"), 1, 2)
df = df.loc[df['filename'] != False]

# shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

np_subsection = [["SCAN_ID", "AGE_M", "SEX", "filename"]]

# random select N_CASES per age group
for year in range(AGE_FROM, AGE_TO):
	subsection = df.loc[df["Age at Diagnosis"] == year][:N_CASES].to_numpy()
	np_subsection = np.concatenate([np_subsection, subsection], axis=0)


# write to the folder selected records and resulting .csv
for i in range(1, np.shape(np_subsection)[0]):
    shutil.copyfile(storage_path + "/" + np_subsection[i, 3],
                    path_for_images + "/" + np_subsection[i, 3])

# write csv
df = pd.DataFrame(np_subsection)
df.to_csv(path_or_buf= path_for_images + "/test_cbtn.csv", header=False, index=False)
