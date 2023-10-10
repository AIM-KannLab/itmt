from __future__ import generators
import logging
import glob, os, functools
import sys
sys.path.append('../TM2_segmentation')
import SimpleITK as sitk
from scipy.signal import medfilt
import numpy as np
from numpy import median
import scipy
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage
from skimage.transform import resize,rescale
import cv2
import itk
import subprocess
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.zscore import ZScoreNormalize

import pandas as pd
import tensorflow as tf
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 
  
from compute_population_pred import major_voting, measure_tm
from scripts.densenet_regression import DenseNet
from scripts.unet import get_unet_2D
from scripts.preprocess_utils import load_nii,save_nii, find_file_in_path,iou, register_to_template,enhance_noN4,crop_center, get_id_and_path
from scripts.feret import Calculater
from settings import  target_size_dense_net, target_size_unet, unet_classes, softmax_threshold, major_voting,scaling_factor
from scripts.infer_selection import get_slice_number_from_prediction, funcy
import warnings

warnings.filterwarnings('ignore')

# MNI templates 
age_ranges = {"data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# function to select the correct MRI template based on the age  
def select_template_based_on_age(age):
    for golden_file_path, age_values in age_ranges.items():
        if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
            #print(golden_file_path)
            return golden_file_path
   
# register the MRI to the template     
def register_to_template(input_image_path, output_path, fixed_image_path,rename_id,create_subfolder=True):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('data/golden_image/mni_templates/Parameters_Rigid.txt')

    if "nii" in input_image_path and "._" not in input_image_path:
        #print(input_image_path)

        # Call registration function
        try:        
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            itk.imwrite(result_image, output_path+"/"+rename_id+".nii.gz")
                
            print("Registered ", rename_id)
        except:
            print("Cannot transform", rename_id)
 
# function to filter the islands in the segmentation mask, to keep only the largest one      
def filter_islands(muscle_seg):
    img = muscle_seg.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(img.shape, np.uint8)
    cnt_mask = np.zeros(img.shape, np.uint8)
    area = 0
    c=0
    print(len(contours))

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(c)
        mask = cv2.fillPoly(mask, pts=[c], color=(255, 0, 0))
        cnt_mask =  cv2.drawContours(cnt_mask, [c], -1, (255, 255, 255), 0)
    
    #save image w contours
    #cv2.imwrite('/media/sda/Anna/TM2_segmentation/data/reg_retest_csv/_retest/contour.png',
    #            cv2.drawContours(cnt_mask, [c], -1, (255, 255, 255), 0))
    #save mask
    #cv2.imwrite('/media/sda/Anna/TM2_segmentation/data/reg_retest_csv/_retest/mask.png',mask)
    #print area of  contour
    #save image with contour
    #plt.imsave('/media/sda/Anna/TM2_segmentation/data/reg_retest_csv/_retest/contour.png',cnt_mask)
    return mask, area, c

                
# this is a retest of the registration process
# this should be run 10 times

output_path = 'data/reg_retest_csv/'
number_of_tests = 10

img_path_list = ['data/reg_retest_csv/test300/C34317_1_T1W/registered_z.nii']
age_list = [9]

#age_list = [9.0]*number_of_tests
#img_path_list = ['docker/data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz']*number_of_tests
print(img_path_list)

model_weight_path_selection = 'model/densenet_models/test/itmt1.hdf5'
model_weight_path_segmentation = 'model/unet_models/test/itmt1.hdf5'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
results=[]

threshold = 0.75
alpha = 0.8 
# load models

model_selection = DenseNet(img_dim=(256, 256, 1), 
                    nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                    compression_rate=0.5, sigmoid_output_activation=True, 
                    activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
model_selection.load_weights(model_weight_path_selection)
print('\n','\n','\n','loaded:' ,model_weight_path_selection)  
        
model_unet = get_unet_2D(unet_classes,(target_size_unet[0], target_size_unet[1], 1),\
            num_convs=2,  activation='relu',
            compression_channels=[16, 32, 64, 128, 256, 512],
            decompression_channels=[256, 128, 64, 32, 16])
model_unet.load_weights(model_weight_path_segmentation)
print('\n','\n','\n','loaded:' ,model_weight_path_segmentation)  
    
for i in range(len(age_list)):
    print("Test",i)
    path_to = output_path+'_retest/'
    if not os.path.exists(path_to):
        os.mkdir(path_to)
    try:
        image, affine = load_nii(img_path_list[i])

        # path to store registered image in
        patient_id = img_path_list[i].split("/")[-1].split(".")[0]
        new_path_to = path_to+patient_id
        if not os.path.exists(new_path_to):
            os.mkdir(new_path_to)

        
        image_sitk = sitk.ReadImage(img_path_list[i])    
        windowed_images  = sitk.GetArrayFromImage(image_sitk)           

        # resize image to 256x256
        resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                                    preserve_range=True, anti_aliasing=True, mode='constant')
        series = np.dstack([resize_func(im) for im in windowed_images])
        series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
        series_n = []

        # create MIP of 5 slices = 5mm
        for slice_idx in range(2, np.shape(series)[0]-2):
            im_array = np.zeros((256, 256, 1, 5))
            
            im_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.float32)
            im_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.float32)
            im_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.float32)
            im_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.float32)
            im_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.float32)
                    
            im_array= np.max(im_array, axis=3)
                    
            series_n.append(im_array)
            series_w = np.dstack([funcy(im) for im in series_n])
            series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])
            
        # predict slice  
        predictions = model_selection.predict(series_w)
        slice_label = get_slice_number_from_prediction(predictions)
        print("Predicted slice:", slice_label)

        img = nib.load(img_path_list[i])  
        image_array, affine = img.get_fdata(), img.affine
        infer_seg_array_3d_1,infer_seg_array_3d_2 = np.zeros(image_array.shape),np.zeros(image_array.shape)
        
        # rescale image into 512x512 for unet 
        image_array_2d = rescale(image_array[:,15:-21,slice_label], scaling_factor).reshape(1,target_size_unet[0],target_size_unet[1],1) 
        
        # create 4 images - half TMT and half empty           
        img_half_11 = np.concatenate((image_array_2d[:,:256,:,:],np.zeros_like(image_array_2d[:,:256,:,:])),axis=1)
        img_half_21 = np.concatenate((np.zeros_like(image_array_2d[:,:256,:,:]),image_array_2d[:,:256,:,:]),axis=1)
        img_half_12 = np.concatenate((np.zeros_like(image_array_2d[:,256:,:,:]),image_array_2d[:,256:,:,:]),axis=1)
        img_half_22 = np.concatenate((image_array_2d[:,256:,:,:],np.zeros_like(image_array_2d[:,256:,:,:])),axis=1)

        flipped = np.flip(image_array_2d, axis=1)

        flipped_11 = np.concatenate((flipped[:,:256,:,:],np.zeros_like(flipped[:,:256,:,:])),axis=1)
        flipped_21 = np.concatenate((np.zeros_like(flipped[:,:256,:,:]),flipped[:,:256,:,:]),axis=1)
        flipped_12 = np.concatenate((np.zeros_like(flipped[:,256:,:,:]),flipped[:,256:,:,:]),axis=1)
        flipped_22 = np.concatenate((flipped[:,256:,:,:],np.zeros_like(flipped[:,256:,:,:])),axis=1)

        list_of_left_muscle = [img_half_11, img_half_21, flipped_12, flipped_22]
        list_of_right_muscle = [img_half_12,img_half_22, flipped_11, flipped_21]

        list_of_left_muscle_preds = []
        list_of_right_muscle_preds = []

        # predict left and right muscle on each of 4 images
        for image in list_of_left_muscle: 
            infer_seg_array = model_unet.predict(image)
            muscle_seg = infer_seg_array[:,:,:,1].reshape(1,target_size_unet[0],target_size_unet[1],1)               
            list_of_left_muscle_preds.append(muscle_seg)
                            
        for image in list_of_right_muscle: 
            infer_seg_array = model_unet.predict(image)
            muscle_seg = infer_seg_array[:,:,:,1].reshape(1,target_size_unet[0],target_size_unet[1],1)             
            list_of_right_muscle_preds.append(muscle_seg)
                        
        list_of_left_muscle_preds_halved = [list_of_left_muscle_preds[0][:,:256,:,:],
                                            list_of_left_muscle_preds[1][:,256:,:,:],
                                            np.flip(list_of_left_muscle_preds[2][:,256:,:,:],axis=1),
                                            np.flip(list_of_left_muscle_preds[3][:,:256,:,:],axis=1)]

        list_of_right_muscle_preds_halved = [list_of_right_muscle_preds[0][:,256:,:,:],
                                            list_of_right_muscle_preds[1][:,:256,:,:],
                                            np.flip(list_of_right_muscle_preds[2][:,:256,:,:],axis=1),
                                            np.flip(list_of_right_muscle_preds[3][:,256:,:,:],axis=1)]
        
        # average predictions and threshold              
        left_half_result = np.mean(list_of_left_muscle_preds_halved, axis=0)<=threshold # <>
        right_half_result = np.mean(list_of_right_muscle_preds_halved, axis=0)<=threshold # <>
        muscle_seg_1 = np.concatenate((left_half_result,np.zeros_like(left_half_result)),axis=1)
        muscle_seg_2 = np.concatenate((np.zeros_like(left_half_result),right_half_result),axis=1)

        infer_seg_array_3d_1_filtered,infer_seg_array_3d_2_filtered = np.zeros(image_array.shape),np.zeros(image_array.shape)
        infer_seg_array_3d_merged_filtered =  np.zeros(image_array.shape)
                
        # filter islands
        
        muscle_seg_2_filtered, area_2, cnt_2 = filter_islands(muscle_seg_2[0])
        muscle_seg_1_filtered, area_1, cnt_1 = filter_islands(muscle_seg_1[0])

        # save plots 
        fg = plt.figure(figsize=(5, 5), facecolor='k')
        I = cv2.normalize(image_array_2d[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(path_to+"/"+patient_id+"_no_masks.png", I)
        im = cv2.imread(path_to+"/"+patient_id+"_no_masks.png")                        
        im_copy = im.copy()        
        result = im.copy()
        for cont in [cnt_1,cnt_2]: 
            if len(cont)!=0:
                if cv2.contourArea(cont) <= 1:
                    im_copy = cv2.drawContours(im_copy, [cont], -1, (0, 0, 255), -1)
                else:
                    im_copy = cv2.drawContours(im_copy, [cont], -1, (51, 197, 255), -1)
        filled = cv2.addWeighted(im, alpha, im_copy, 1-alpha, 0)
        for cont in [cnt_1,cnt_2]: 
            if len(cont)!=0:
                if cv2.contourArea(cont) <= 1:
                    result = cv2.drawContours(filled, [cont], -1, (0, 0, 255), 0)
                else:
                    result = cv2.drawContours(filled, [cont], -1, (51, 197, 255), 0)

        cv2.imwrite(path_to+"/"+patient_id+"_mask.png", result)
                
        # rescale for the unet
        infer_seg_array_2d_1 = rescale(muscle_seg_1[0],1/scaling_factor)
        infer_seg_array_2d_2 = rescale(muscle_seg_2[0],1/scaling_factor)
        infer_seg_array_2d_1_filtered = rescale(muscle_seg_1_filtered,1/scaling_factor)
        infer_seg_array_2d_2_filtered = rescale(muscle_seg_2_filtered,1/scaling_factor)

        # save to 3d
        infer_seg_array_3d_1[:,:,slice_label] = np.pad(infer_seg_array_2d_1[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
        infer_seg_array_3d_2[:,:,slice_label] = np.pad(infer_seg_array_2d_2[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
        infer_seg_array_3d_1_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_1_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
        infer_seg_array_3d_2_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_2_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
                    
        concated = np.concatenate((infer_seg_array_2d_1_filtered[:100,:,0],infer_seg_array_2d_2_filtered[100:,:,0]),axis=0)    
        infer_seg_array_3d_merged_filtered[:,:,slice_label] = np.pad(concated,[[0,0],[15,21]],'constant',constant_values=0)

        ## calculate TM          
        infer_seg_array_2d_1 = rescale(muscle_seg_1[0],1/scaling_factor)
        infer_seg_array_2d_2 = rescale(muscle_seg_2[0],1/scaling_factor)

        infer_seg_array_3d_1_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_1_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
        infer_seg_array_3d_2_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_2_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)

        objL_pred_minf_line, objR_pred_minf_line, objL_pred_minf, objR_pred_minf = 0,0,0,0
                        
        
        if np.sum(infer_seg_array_3d_1_filtered[:100,:,slice_label])>2:
            objL_pred_minf = round(Calculater(infer_seg_array_3d_1_filtered[:100,:,slice_label], edge=True).minf,2)
            Calculater(infer_seg_array_3d_1_filtered[:100,:,slice_label], edge=True).plot()
            plt.savefig(path_to+"/"+patient_id+"_left.png")

        if np.sum(infer_seg_array_3d_2_filtered[100:,:,slice_label])>2:
            objR_pred_minf = round(Calculater(infer_seg_array_3d_2_filtered[100:,:,slice_label], edge=True).minf,2)
            Calculater(infer_seg_array_3d_2_filtered[100:,:,slice_label], edge=True).plot()
            plt.savefig(path_to+"/"+patient_id+"_right.png")
                    
        CSA_PRED_TM1 = np.sum(infer_seg_array_3d_1_filtered[:100,:,slice_label])
        CSA_PRED_TM2 = np.sum(infer_seg_array_3d_2_filtered[100:,:,slice_label])
                            
        
        input_tmt = (objL_pred_minf+objR_pred_minf)/2
        print("Age:",str(age_list[i]),"iTMT[mm]:", input_tmt)   
        
        
        results.append(['test'+str(i),patient_id,str(round(float(age_list[i]),2)),
                str(slice_label), str(input_tmt)])
    except:
        print("Cannot process", img_path_list[i])
        continue
    
    
# error found: was taking unfitered mask for test; check if that is the case for all tests    