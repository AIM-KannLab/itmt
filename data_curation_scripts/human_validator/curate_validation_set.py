import pandas as pd
import numpy as np

import sys
# setting path
sys.path.append('../TM2_segmentation')

input_annotation_file = 'data/pop_norms.csv'   
df = pd.read_csv(input_annotation_file, header=0)
df = df.sample(frac=1).reset_index(drop=True)

N = 50

counts_per_age =[[] for x in range(35-4+1)] 

for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    if len(counts_per_age[row['Age']-4])<N:
        counts_per_age[row['Age']-4].append([row["ID"],row["Age"],row["Gender"],row['Dataset']])
    
print(len(counts_per_age))

lst_counts = []

for i in range(0,len(counts_per_age)):
    print(i+4,len(counts_per_age[i]))
    for j in counts_per_age[i]:
        lst_counts.append(j)
    
## save to the new dataframe
df = pd.DataFrame(lst_counts)
df.to_csv(path_or_buf= "data/dataset_validation.csv", header=['ID','Age','Gender','Dataset'])
   
