import os
import dicom2nifti 
# unzip first
import tarfile
# importing the zipfile module
from zipfile import ZipFile
import shutil
# open file
import subprocess

input_path = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/t1/"
extracted_path ='/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/'
path_to_save_nii = '/mnt/kannlab_rfa/Anna/PING/Package_1207895/niftis/'

#/Volumes/BWH-KANNLAB/Anna/PING/Package_1207895/image03/extracted_t1/
# P0007/P0007_t1/data/ping_dicoms/files/P0007_000_02/scans/2-IRSPGR_PROMO/resources/DICOM/files

#(base) anna@PHS026891 P0016_000_01 % pwd
#/Volumes/BWH-KANNLAB/Anna/PING/Package_1207895/image03/extracted_t1/P0016_t1/
# P0016/P0016_t1/data/ping_dicoms/files/P0016_000_01
for i in range(0,len(os.listdir("/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1"))):
    filename = os.listdir("/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1")[i]
    if ".DS" not in filename and "_t1" in filename:
        #print(filename)
        for pr in os.listdir("/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/"+filename +"/" + filename.split("_")[0]+"/" + filename +"/data/ping_dicoms/files/"):
            print(pr)
            dst_dir = extracted_path+pr+"/"
            src_dir = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/"+filename +"/" + filename.split("_")[0]+"/" + filename +"/data/ping_dicoms/files/" + pr +"/"
            print(src_dir,dst_dir)
            os.system('cp -R '+src_dir+' '+dst_dir)  
            #shutil.rmtree(src_dir)
            #subprocess.check_output(['rm', '-rf', src_dir])
            #shutil.copytree(src_dir, dst_dir)
            #break        