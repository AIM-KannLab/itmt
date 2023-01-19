import os
import dicom2nifti 
# unzip first
import tarfile
# importing the zipfile module
from zipfile import ZipFile

# open file

input_path = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/t1/"
extracted_path ='/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/'
path_to_save_nii = '/mnt/kannlab_rfa/Anna/PING/Package_1207895/niftis/'

#/Volumes/BWH-KANNLAB/Anna/PING/Package_1207895/image03/extracted_t1/
# P0007/P0007_t1/data/ping_dicoms/files/P0007_000_02/scans/2-IRSPGR_PROMO/resources/DICOM/files
for i in range(193,194):#len(os.listdir("/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/"))//2):
    filename = os.listdir("/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/")[i]
    print(i, filename)
    if ".DS_Store" not in filename and "_t1" not in filename:
        ''' unzip_path = extracted_path+filename.split(".")[0] 
        os.mkdir(unzip_path)
        print(unzip_path)
        
        # loading the temp.zip and creating a zip object
        with ZipFile(input_path+filename, 'r') as zObject:
            zObject.extractall(path=unzip_path)
        '''   
        #copy_paths.py
        
        path_to_dicom_1 = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/" + filename + '/scans/'
        for d in os.listdir(path_to_dicom_1):
            if ".DS_Store" not in d and "IRSPGR_PROMO" in d:
                print(d)
                try:
                    path_to_dicom_1_1 = path_to_dicom_1 + d + '/resources/DICOM/files'
                    path_to_save_nii_file1 = path_to_save_nii + "IRSPGR_PROMO/"+ filename.split(".")[0] +".nii.gz"
                    print(path_to_dicom_1_1, path_to_save_nii_file1)
                    
                    #dicom2nifti.dicom_series_to_nifti(path_to_dicom_1_1,path_to_save_nii_file1, reorient_nifti=True)
                except:
                    pass
            
        path_to_dicom_2 = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/" + filename + '/scans/'
        for d in os.listdir(path_to_dicom_2):
            if ".DS_Store" not in d and "SAG_CUBE_PROMO" in d:
                try:
                    path_to_dicom_2_2 = path_to_dicom_2 + d + '/resources/DICOM/files'
                    path_to_save_nii_file2 = path_to_save_nii + "SAG_CUBE_PROMO/"+ filename.split(".")[0] +".nii.gz"
                    #dicom2nifti.dicom_series_to_nifti(path_to_dicom_2_2,path_to_save_nii_file2, reorient_nifti=True)
                except:
                    pass
                
        path_to_dicom_3 = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/"+filename+"/"+filename+"/scans/"
        if os.path.exists(path_to_dicom_3):
            for d in os.listdir(path_to_dicom_3):
                if ".DS_Store" not in d and "MPRAGE" in d: 
                    try:
                        path_to_dicom_3_3 = path_to_dicom_3 + d + '/resources/DICOM/files'
                        path_to_save_nii_file3 = path_to_save_nii + "MPRAGE/"+ filename.split(".")[0] +".nii.gz"
                        #dicom2nifti.dicom_series_to_nifti(path_to_dicom_3_3,path_to_save_nii_file3, reorient_nifti=True)
                    except:
                        pass
                    
        path_to_dicom_4 = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/"+filename+"/scans/"
        if os.path.exists(path_to_dicom_4):
            for d in os.listdir(path_to_dicom_4):
                if ".DS_Store" not in d and "MPR" in d: 
                    #print(d)
                    try:
                        path_to_dicom_4_4 = path_to_dicom_4 + d + '/resources/DICOM/files'
                        path_to_save_nii_file4 = path_to_save_nii + "MPRAGE/"+ filename.split(".")[0] +".nii.gz"
                        #dicom2nifti.dicom_series_to_nifti(path_to_dicom_4_4,path_to_save_nii_file4, reorient_nifti=True)
                    except:
                        pass
        
        #extracted_t1/"+filename+"/"+filename+"/scans/401-MPRAGE/resources/DICOM/files
        #IRSPGR_PROMO + MPRAGE is non ench - the one we need
 