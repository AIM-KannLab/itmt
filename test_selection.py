import argparse
import warnings
import os 
from scripts.infer_selection import test
from settings import CUDA_VISIBLE_DEVICES

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")   

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description=' Top slice is selected by AI algorithm.') 

    parser.add_argument('--image_dir','-i', type = str, default = 'data/z_scored_mris/z_with_pseudo/z/',
                        help = 'Directory path of scan')
    parser.add_argument('--model_weight_path','-m', type = str, default = 'model/densenet_models/test/brisk-pyramid.hdf5', #'model/densenet_models/train/untitled_model_2023_07_11_14_24_45/Top_Selection_Model_Weight.hdf5',   #'model/densenet_models/test/itmt1.hdf5', 
                        help = 'File path of well-trained model weight')
    parser.add_argument('--csv_write_path','-c', type = str, default = 'data/curated_test/final_test/Cancer_Slice_Prediction2.csv',
                        help = 'File path of csv file to write ')
    parser.add_argument('--input_annotation_file','-a', type = str, default = 'data/all_metadata.csv',
                        help = 'File path of metadata')
    
    args = parser.parse_args()
    model = test(**vars(args))