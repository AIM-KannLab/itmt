from __future__ import generators
import logging
import glob, os, functools, gc
import pandas as pd
import numpy as np
import sys
import math
from dateutil import parser
import itertools
sys.path.append('../TM2_segmentation')


def isNaN(num):
    if type(num)==str:
        return False
    if float('-inf') < float(num) < float('inf'):
        return False 
    else:
        return True
    
list_of_extracted_labs = []
requested_lab = 'Hemoglobin'
input_path = 'data/labs/'
# if Column1 in 0,0, delete the column
# delete first row
# parse dates until the text metadata is found

#labs_path = 'data/labs/198938_labs.csv'
#bmi_path = 'data/labs/4255354_bmi.csv'
for labs_path in glob.glob(input_path+'*labs.csv'):
    patient_id = labs_path.split('_')[0].split('/')[-1]
    print(patient_id)
    # read the file
    df=pd.read_csv(labs_path, sep=',',header=None)
    
    # delete first row
    df.drop(df.index[0], inplace=True)
    
    # parse dates until the text metadata is found
    df_storage = []
    sub_df=[]
    for i in range(0, len(df)): 
        try:
            converted_date  = parser.parse(df.iloc[i,0])   
        except:
            # if text metadata found -> convert sub_df to dataframe
            df_storage.append(sub_df)
            sub_df = [] 
            sub_df.append(list(df.iloc[i]))
            
        else:
            # if no exception, add to sub_df
            sub_df.append(list(df.iloc[i]))
        
    # first element is empty in df_storage, delete it
    df_storage.pop(0)       
    df_storage_converted = []

    for list_to_convert in df_storage:
        column_names = np.asarray(list_to_convert[0])         
        df_converted = pd.DataFrame(list_to_convert, columns = column_names)
        
        # delete the first row
        df_converted.drop(df_converted.index[0], inplace=True)
        df_converted = df_converted.reset_index(drop=True)
        df_converted.rename(columns={ df_converted.columns[0]: "time" }, inplace = True)
        
        # Columns to not rename
        excluded = df_converted.columns[~df_converted.columns.duplicated(keep=False)]

        # An incrementer
        inc = itertools.count().__next__

        # A renamer
        def ren(name):
            return f"{name}{inc()}" if name not in excluded else name

        # Use inside rename()
        df_converted.rename(columns=ren, inplace=True)
        
        ## column names are the names of the metrics
        ## first column is the date
        ## save to the new dataframe patient id from the name of the file, and the date, and the metric from requested_lab
        for column_name in list_to_convert[0]:
            if requested_lab==column_name:
                for i in range(0, len(df_converted)):
                    try:
                        if isNaN(df_converted[requested_lab][i]):
                            continue
                    except:
                        continue
                    else:
                        list_of_extracted_labs.append([patient_id,
                                                    df_converted[df_converted.columns[0]][i],
                                                    df_converted[requested_lab][i]])
pd.DataFrame(list_of_extracted_labs, columns=['patient_id','date',requested_lab]).to_csv('itmt_v2.0/'+requested_lab+'.csv', index=False)