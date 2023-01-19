import pandas as pd
import numpy as np

import glob, os, functools
from skimage.transform import resize,rescale
import sys
import datetime
import json
import shutil
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
import tensorflow as tf

# setting path
sys.path.append('../TM2_segmentation')

from settings import  target_size_dense_net, target_size_unet, unet_classes, softmax_threshold, major_voting,scaling_factor
from scripts.generators import SliceSelectionSequence
from scripts.densenet_regression import DenseNet
from sklearn.model_selection import train_test_split
from scripts.infer_selection import get_slice_number_from_prediction, funcy

dict_paths = {
    'ABCD': "data/t1_mris/registered",
    'ABIDE': "data/t1_mris/abide_ench_reg",
    'AOMIC':"data/t1_mris/aomic_reg_ench",
    'Baby': "data/t1_mris/registered",
    'Calgary': "data/t1_mris/calgary_reg_ench",
    'ICBM': "data/t1_mris/icbm_ench_reg",
    'IXI':"data/t1_mris/registered",
    'NIMH': "data/t1_mris/nihm_ench_reg",
    'PING': "data/t1_mris/pings_ench_reg",
    'Pixar': 'data/t1_mris/pixar_ench',
    'SALD': "data/t1_mris/sald_reg_ench",
    'NYU': "data/t1_mris/nyu_reg_ench",
    'Healthy adults': "data/t1_mris/healthy_adults_nihm_reg_ench",
    'Petfrog':"data/t1_mris/petfrog_reg_ench"}

def find_path(img_id, dataset):
    dataset_image_path = dict_paths[dataset]
    for image_path in os.listdir(dataset_image_path+"/no_z"):
        if img_id in image_path:
            image_path_no_z = dataset_image_path+"/no_z/"+image_path
            image_path_z = dataset_image_path+"/z/"+img_id+"/"+img_id+".nii"
    patient_id = str(img_id).split(".")[0]  
    return image_path_z, image_path_no_z, patient_id

input_annotation_file =  "data/dataset_validation.csv"  
save_pics_to = "data_curation_scripts/human_validator/pics/"
save_niftys_to = "data_curation_scripts/human_validator/raw_niftys_pics/"

## since i did not store the slice prediction and realised it too late;
# create new slice prection and store it in the file 

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
#path to ench and registered file
#image_dir = 'data/t1_mris/cbtn_reg_ench/z/' 
model_weight_path_selection = 'model/densenet_models/test/brisk-pyramid.hdf5'

# load models
model_selection = DenseNet(img_dim=(256, 256, 1), 
                nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                compression_rate=0.5, sigmoid_output_activation=True, 
                activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
model_selection.load_weights(model_weight_path_selection)
print('\n','\n','\n','loaded:' ,model_weight_path_selection)  


df = pd.read_csv(input_annotation_file, header=0)

mathching_imgs = []

for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    image_path_z, image_path_no_z,patient_id = find_path(row['ID'], row['Dataset'])
    print(image_path_z, image_path_no_z,patient_id)
    
    if len(image_path_z)>3:
        image_sitk = sitk.ReadImage(image_path_z)    
        windowed_images  = sitk.GetArrayFromImage(image_sitk)   
                
        # enchancing is done in the preprocessing step
        resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                            preserve_range=True, anti_aliasing=True, mode='constant')
        series = np.dstack([resize_func(im) for im in windowed_images])
        series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])

        series_n = []

        for slice_idx in range(2,np.shape(series)[0]-2):
            im_array = np.zeros((256,256,1,5))
                
            im_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.float32)
            im_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.float32)
            im_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.float32)
            im_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.float32)
            im_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.float32)
                
            im_array = np.max(im_array, axis=3)
                
            series_n.append(im_array)
            series_w = np.dstack([funcy(im) for im in series_n])
            series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])

        predictions = model_selection.predict(series_w)
        slice_label = get_slice_number_from_prediction(predictions)
        
        img = nib.load(image_path_no_z)  
        image_array, affine = img.get_fdata(), img.affine
        image_array_2d = rescale(image_array[:,15:-21,slice_label], scaling_factor).reshape(1,target_size_unet[0],target_size_unet[1],1) 
        
        fig=plt.figure(figsize=(15,6))
        plt.imshow(image_array_2d[0], cmap='gray', interpolation='none')
        plt.savefig(save_niftys_to+patient_id+".png")
        
        #search for the images in the  and save to save_pics_to
        lst_w_segments =[]
        for img in os.listdir('data/t1_mris/pics'):
            if patient_id in img:
                shutil.copyfile('data/t1_mris/pics/' + img,save_pics_to + img)
                lst_w_segments.append(save_pics_to + img)
            
        if len(lst_w_segments)>2:
            print(row['ID'],lst_w_segments)
        elif len(lst_w_segments)==2:
            mathching_imgs.append([row['ID'], row['Dataset'],row['Age'],row['Gender'],
                               save_niftys_to+patient_id+".png",
                               lst_w_segments[0],lst_w_segments[1]
                               ])
        #break
    
df = pd.DataFrame(mathching_imgs)
df.to_csv(path_or_buf= "data/dataset_validation_w_imgs.csv", header=['ID','Dataset','Age','Gender',
                                                                     'Raw_path',"Segmentation_1","Segmentation_2"])

#conda activate tf2_py39       
#python data_curation_scripts/human_validator/collect_pics_and_niftys.py
