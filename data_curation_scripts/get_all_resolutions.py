import sys
# setting path
sys.path.append('../TM2_segmentation')

import os
import numpy as np
import nibabel as nib
import itk
import pandas as pd
import tarfile


#input_annotation_file = 'data/all_metadata.csv'  
#df = pd.read_csv(input_annotation_file, header=0,delimiter=',')
input_path = 'data/curated_test/raw_curation/'#"/mnt/aertslab/USERS/Anna/TM-segmentation/data/pre-processing/test"

final_metadata = []
for file in os.listdir(input_path):
    if "._" not in file and  ".nii" in file: 
        n1_img = nib.load(input_path+file)
        n1_header = n1_img.header  
        final_metadata.append([file,
                n1_header.get_base_affine()[0,0],
                n1_header.get_base_affine()[1,1],
                n1_header.get_base_affine()[2,2],
                n1_header.get_base_affine()[0,2],
                n1_header.get_base_affine()[1,2],
                n1_header.get_base_affine()[2,2]])

df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/test_resolutions3.csv",header = ['Filename','x','y','z','x_rot','y_rot','z_rot'])

