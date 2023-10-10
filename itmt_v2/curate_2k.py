import sys
sys.path.insert(1, '../')

import numpy as np
import pandas as pd
import os
import shutil

dataset_dict_paths = {
    'ABCD': "/media/sda/Anna/TM2_segmentation/data/t1_mris/registered/z/",
    'ABIDE': "/media/sda/Anna/TM2_segmentation/data/t1_mris/abide_ench_reg/z/",
    'AOMIC':"/media/sda/Anna/TM2_segmentation/data/t1_mris/aomic_reg_ench/z/",
    'Baby': "/media/sda/Anna/TM2_segmentation/data/t1_mris/registered/z/",
    'Calgary': "/media/sda/Anna/TM2_segmentation/data/t1_mris/calgary_reg_ench/z/",
    'ICBM': "/media/sda/Anna/TM2_segmentation/data/t1_mris/icbm_ench_reg/z/",
    'IXI':"/media/sda/Anna/TM2_segmentation/data/t1_mris/registered/z/",
    'NIMH': "/media/sda/Anna/TM2_segmentation/data/t1_mris/nihm_ench_reg/z/",#
    'PING': "/media/sda/Anna/TM2_segmentation/data/t1_mris/pings_ench_reg/z/",
    'Pixar': '/media/sda/Anna/TM2_segmentation/data/t1_mris/pixar_ench/z/',
    'SALD': "/media/sda/Anna/TM2_segmentation/data/t1_mris/sald_reg_ench/z/",
    'NYU': "/media/sda/Anna/TM2_segmentation/data/t1_mris/nyu_reg_ench/z/",
    'Healthy adults': "/media/sda/Anna/TM2_segmentation/data/t1_mris/healthy_adults_nihm_reg_ench/z/",#
    'Petfrog':"/media/sda/Anna/TM2_segmentation/data/t1_mris/petfrog_reg_ench/z/", #
    '28':'/media/sda/Anna/TM2_segmentation/data/t1_mris/28_reg_ench/z/',
    "BCH":'/media/sda/Anna/TM2_segmentation/data/t1_mris/bch_reg_ench/z/',
    "DMG":'/media/sda/Anna/TM2_segmentation/data/t1_mris/dmg_reg_ench/z/',
    "long579":'/media/sda/Anna/TM2_segmentation/data/t1_mris/long579_reg_ench/z/',
    'pseudo':'/media/sda/Anna/TM2_segmentation/data/z_scored_mris/z_with_pseudo/z/',
    }

input_annotation_file = 'itmt_v2.0/itmt2.0.csv'
#'../private_notebooks/dataset_for_retraining.csv'
df = pd.read_csv(input_annotation_file, header=0)
#print(df)
#ID,Dataset,Age,Gender

# cp data/pseudolabels_registered/ data/itmt2.0/
output_path = 'data/itmt2.0/'
segmentations_dir = 'data/t1_mris/'
old_segmentations = '/media/sda/Anna/TM2_segmentation/data/pseudolabels_registered/'

new_df_list = []
for i in range(len(df)):
    dataset = dataset_name = df['Dataset'].iloc[i]
    if dataset=='NIMH':
        dataset_name = 'HIMH'
    if dataset=='Healthy adults' or dataset=='Petfrog':
        dataset_name = 'HAN'
    if dataset=='Baby':
        dataset_name = 'BABY'
        
    patient_id = df['Filename'].iloc[i].split(".")[0]
    infer_3d_path = segmentations_dir+"/pics/niftis/"+str(dataset_name)+"_"+patient_id + '_AI_seg.nii.gz'
    
    if dataset_name == 'HIMH' and 'deface' in patient_id:
        #infer_3d_path = 'data/z_scored_mris/z_with_pseudo/z/'+patient_id + '/TM.nii.gz'
    
        #os.mkdir(output_path+patient_id.split(".")[0])
        #shutil.copy(infer_3d_path, output_path+patient_id.split(".")[0]+'/TM.nii.gz')
        #shutil.copy(dataset_dict_paths['pseudo']+patient_id+"/"+patient_id+'.nii', output_path+patient_id.split(".")[0]+'/'+patient_id+'.nii.gz')
            
        new_df_list.append([patient_id+'.nii.gz',df['AGE_M'].iloc[i],df['SEX'].iloc[i],dataset_name])
    elif 'IXI' in patient_id:
        infer_3d_path = 'data/z_scored_mris/z_with_pseudo/z/'+patient_id + '/TM.nii.gz'
    
        os.mkdir(output_path+patient_id.split(".")[0])
        shutil.copy(infer_3d_path, output_path+patient_id.split(".")[0]+'/TM.nii.gz')
        shutil.copy(dataset_dict_paths['pseudo']+patient_id+"/"+patient_id+'.nii', output_path+patient_id.split(".")[0]+'/'+patient_id+'.nii.gz')
            
        new_df_list.append([patient_id+'.nii.gz',df['AGE_M'].iloc[i],df['SEX'].iloc[i],dataset_name])
    else:
        try:
            #os.mkdir(output_path+patient_id.split(".")[0])
            #shutil.copy(infer_3d_path, output_path+patient_id.split(".")[0]+'/TM.nii.gz')
            #shutil.copy(dataset_dict_paths[dataset]+patient_id+"/"+patient_id+'.nii', output_path+patient_id.split(".")[0]+'/'+patient_id+'.nii.gz')
            new_df_list.append([patient_id+'.nii.gz',df['AGE_M'].iloc[i],df['SEX'].iloc[i],dataset_name])
        except:
            continue
            
pd.DataFrame(new_df_list).to_csv('data/itmt2.0/itmt2.0.csv',header=['Filename','AGE_M','SEX','Dataset'],index=False)
    #break
    #if  os.path.exists(dataset_dict_paths[dataset]+patient_id+"/"+patient_id+'.nii')==False:
    #    print(df['ID'].iloc[i],dataset_dict_paths[dataset],dataset,
    #      os.path.exists(dataset_dict_paths[dataset]+patient_id+"/"+patient_id+'.nii'))
            

