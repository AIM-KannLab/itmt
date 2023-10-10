import argparse
import os
import warnings
from itmt_v2.infer_segmentation import test
from settings import CUDA_VISIBLE_DEVICES

os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")
   
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bodycomposition is segmented and stored in NIFTI format.') 
    
    parser.add_argument('--image_dir','-i', type = str, #default= 'data/z_scored_mris/z_with_pseudo/z/',
                        default='data/curated_test/final_test/z/',
                        help = 'File path of Mris scans')
    
    parser.add_argument('--model_weight_path','-m', type = str, default ='model/unet_models/test/itmt1.hdf5', # 'model/unet_models/train/untitled_model_2023_07_15_20_42_53/Top_Segmentation_Model_Weight.hdf5',
                        #'model/unet_models/test/itmt1.hdf5',
                        help = 'File path of well-trained model weight')
    
    parser.add_argument('--slice_csv_path','-c', type = str,# default = 'data/all_metadata.csv',
                        default ='data/curated_test/final_test/Dataset_test_rescaled.csv',
                        help = 'Metadata path')
    
    parser.add_argument('--output_dir','-o', type = str, 
                        default ='itmt_v2/outputs/',
                        help = 'File path to save outputs to')    

    parser.add_argument('--measure_iou','-u', type = bool, 
                        default = True,
                        help = 'To measure IOU of prediced mask and ground truth annotation')  
    
    args = parser.parse_args()
    model = test(**vars(args))
    
