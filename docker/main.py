#!/usr/bin/env python3

import argparse
import os
import warnings
from predict import predict_itmt
from settings import CUDA_VISIBLE_DEVICES

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Input T1.nii.gz to predict iTMT(temporalis muscle thickness)')
    parser.add_argument('--age', '-a',type = float, default = 9.0,
                        help = 'Age of MRI subject in YEARS')
    parser.add_argument('--gender', '-g',type = str, default = 'F',
                        help = 'Gender MRI subject (M/F)')
    parser.add_argument('--img_path', '-pf',type = str, default = 'data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz',
                        help = 'Path to input MRI subject')
    parser.add_argument('--path_to', '-pt',type = str, default = 'data/',
                        help = 'Path to save results')
    parser.add_argument('--cuda_visible_devices', '-c',type = str, default = '0',
                        help = 'Specify cuda visible devices, default:0')
    parser.add_argument('--model_weight_path_selection', '-d',type = str, default = 'model_weights/test/brisk-pyramid.hdf5',
                        help = 'Slice selection model path')
    parser.add_argument('--model_weight_path_segmentation', '-u',type = str, default = 'model_weights/Top_Segmentation_Model_Weight.hdf5',
                        help = 'Segmentation model path')
    parser.add_argument('--df_centile_boys_csv', '-m',type = str, default = 'percentiles_chart_boys.csv',
                        help = 'CSV centiles path, boys model')
    parser.add_argument('--df_centile_girls_csv', '-w',type = str, default = 'percentiles_chart_girls.csv',
                        help = 'CSV centiles path, girls model') 
    
    args = parser.parse_args()
    itmt = predict_itmt(**vars(args))
    
