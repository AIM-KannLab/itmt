import os
from matplotlib import image
import numpy as np
import nibabel as nib
import itk
import pandas as pd
import tarfile
import shutil

from scripts.preprocess_utils import find_file_in_path

input_path = "data/pseudolabels_mris"
# +
for id_patient in os.listdir("data/registered/all_mris"):
    filepath = "data/registered/all_mris/" + id_patient
    image_path = find_file_in_path(id_patient, os.listdir(filepath))
    
    if "Arm" in id_patient:
        final_path = filepath + "/" + id_patient+".nii"
    else:
        final_path = filepath + "/" + id_patient+".nii.gz"
    print(final_path, input_path)
    shutil.copyfile(final_path, input_path + "/" + id_patient+".nii.gz")
    
