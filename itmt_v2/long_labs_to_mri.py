from __future__ import generators
import logging
import glob, os, functools
import gc
import pandas as pd
import numpy as np
import sys
import math
from dateutil import parser
sys.path.append('../TM2_segmentation')

path_to_labs = '/media/sda/Anna/TM2_segmentation/itmt_v2.0/labs_csv/WBC.csv'
df_albumin = pd.read_csv(path_to_labs)
path_to_long_data = '/media/sda/Anna/longitudinal_nifti_BCH/curated_nifti_data/'
found_match_list = []

for i in range(0, len(df_albumin)):
    #match the ids
    patient_id = df_albumin['patient_id'].iloc[i]
    for path_long in os.listdir(path_to_long_data):
        if str(patient_id) == str(path_long):
            #match the date
            for date_study in os.listdir(path_to_long_data+path_long):
                date_diff = (parser.parse(df_albumin['date'].iloc[i])-parser.parse(date_study)).days
                print(patient_id,date_diff,df_albumin['date'].iloc[i],date_study)
                if abs(date_diff)<60:
                    found_match_list.append([patient_id,date_diff,df_albumin['date'].iloc[i],date_study])
                
created_df = pd.DataFrame(found_match_list, columns=['patient_id', 'days_between',
                                        'albumin_date', 'scan_date'])
created_df.to_csv('itmt_v2.0/long_'+path_to_long_data.split("/")[-2]+'_'+path_to_labs.split("/")[-1].split(".")[0]+'_matches.csv', index=False)
print("N patients:", len(created_df['patient_id'].unique()))

            

