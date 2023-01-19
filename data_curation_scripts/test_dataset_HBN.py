import os
import pandas as pd
import numpy as np
import shutil
import subprocess
from settings import AGE_FROM, AGE_TO

N_CASES = 2
path_for_images = "./data/pre-processing/test"

def find_file_in_path(name, path):
    result = False
    for root, dirs, files in os.walk(path):
        result = next((os.path.join(file) for file in files if name in file), False)
    return result

# Sex: 1 = Female, 0 = Male
# read in csv
df = pd.read_csv("./data/HBN_R10_Pheno.csv", delimiter="," , header=0)
df=df[["EID", "Age", "Sex"]]
df["Sex"] = np.where(df["Sex"].str.contains("0"), 1, 2)
df["Age"]=round(df['Age'],0)

np_subsection = [["EID", "Age", "Sex"]]

# random select N_CASES per age group
for year in range(AGE_FROM, AGE_TO):
	subsection = df.loc[df["Age"] == year][:N_CASES].to_numpy()
	np_subsection = np.concatenate([np_subsection, subsection], axis=0)

sites_list = ["Site-SI", "Site-RU", "Site-CUNY", "Site-CBIC"]
for i in range(1,np.shape(np_subsection)[0]):
	# check in which folder case is present
	for site in sites_list:
		duck_line = "duck --username anonymous --list s3:/fcp-indi/data/Projects/HBN/MRI/" + site + "/ "
		output = subprocess.getoutput(duck_line)
		file_id = np_subsection[i,0]
		if file_id in output:
			# download with duck
			duck_line = "duck --username anonymous -d s3:/fcp-indi/data/Projects/HBN/MRI/" +site+ "/sub-" + file_id + "/anat/*vNav_T1w.nii.gz* D://BWH//brain_age_3_datasets_stats//brain-age//eval//"
			print(duck_line)
			output = subprocess.getoutput(duck_line)

# generate csv
df = pd.DataFrame(np_subsection)
df.to_csv(path_or_buf=path_for_images +"/test_hbn.csv", header=False, index=False)

