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

df_albumin = pd.read_csv('/media/sda/Anna/TM2_segmentation/itmt_v2.0/Albumin.csv')
df_scan_date = pd.read_csv('itmt_v2.0/metadata_bch_scan_date.csv')

found_match_list = []
for i in range(0, len(df_albumin)):
    patient_id = df_albumin['patient_id'].iloc[i]
    print(patient_id,df_scan_date[df_scan_date['BCH_MRN']==patient_id]['scan_date'])
    if len(df_scan_date[df_scan_date['BCH_MRN']==patient_id]['scan_date'].values)==0:
        continue
    scan_date = df_scan_date[df_scan_date['BCH_MRN']==patient_id]['scan_date'].values[0]
    if scan_date!='nan':
        found_match_list.append([patient_id,
                                 abs((parser.parse(scan_date)-parser.parse(df_albumin['date'].iloc[i])).days),
                                 df_albumin['Albumin'].iloc[i].split(' ')[0]])
        #print(patient_id, parser.parse(scan_date)-parser.parse(df_albumin['date'].iloc[i]))

matched_df = pd.DataFrame(found_match_list, columns=['patient_id', 'days_between', 'albumin'])
matched_df.to_csv('itmt_v2.0/albumin_scan_date.csv', index=False)
print(matched_df.head())
print("Total matched", len(matched_df['patient_id'].unique()))
print("Total matched", len(matched_df[matched_df['days_between']<100]['patient_id'].unique()))