import argparse
import os
import warnings
from scripts.infer_segmentation import test
from settings import CUDA_VISIBLE_DEVICES

os.environ['CUDA_VISIBLE_DEVICES'] = "0" #CUDA_VISIBLE_DEVICES
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")
   
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bodycomposition is segmented and stored in NIFTI format.') 
    
    parser.add_argument('--image_dir','-i', type = str, default= 'data/curated_test/final_test/z/', #'data/z_scored_mris/z_with_pseudo/z/',# 'data/curated_test/final_test/z/',#'data/z_scored_mris/z_with_pseudo/z/',
                        help = 'File path of Mris scans')
    
    parser.add_argument('--model_weight_path','-m', type = str, default ='model/unet_models/test/Top_Segmentation_Model_Weight.hdf5',
                        help = 'File path of well-trained model weight')
    
    parser.add_argument('--slice_csv_path','-c', type = str, default = 'data/curated_test/final_test/Dataset_test_rescaled.csv',#'data/curated_test/final_test/Dataset_test_rescaled.csv',#'data/all_metadata.csv',
                        help = 'Metadata path')
    
    parser.add_argument('--output_dir','-o', type = str, 
                        default = 'data/curated_test/final_test/predictions_seg_filtered/',#'data/z_segmentations_pseudo/test/',#'data/curated_test/final_test/predictions_seg_filtered/',#'data/z_segmentations_pseudo/test/',
                        help = 'File path to save outputs to')    

    parser.add_argument('--measure_iou','-u', type = bool, 
                        default = True,
                        help = 'To measure IOU of prediced mask and ground truth annotation')  
    
    args = parser.parse_args()
    model = test(**vars(args))
    
