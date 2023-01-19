import argparse
from scripts.segmentation import train
import os
import warnings
from settings import CUDA_VISIBLE_DEVICES

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description  =  'Train a Unet model on multiple classes')
    #the best one is z_segmentations_smaller
    parser.add_argument('--data_dir', '-d',type = str, default = 'data/z_segmentations_pseudo', #'data/z_segmentations_2nd_unet', 
                        help  =  'Directory in which the segmentation training data arrays are stored')
    parser.add_argument('--model_dir', '-m',type = str, default = 'model/unet_models/train', 
                        help = 'Location where trained models are to be stored')
    
    parser.add_argument('--epochs', '-e', type  =  int, default  =  40, help = 'number of training epochs')#40
    parser.add_argument('--batch_size','-b', type  =  int, default  =  4, help = 'batch size')
    parser.add_argument('--load_weights','-w', help  =  'load weights in this file to initialise model')
    parser.add_argument('--name','-a', help  =  'trained model will be stored in a directory with this name')
    parser.add_argument('--gpus','-g', type  =  int, default  =  1, help  =  'number of gpus')
    parser.add_argument('--learning_rate','-l', type = float, default = 5e-4, help = 'learning rate')
    parser.add_argument('--upsamlping_modules','-D', type = int,  default = 5,
                        help = 'downsampling/upsamlping module numbers')
    parser.add_argument('--initial_features','-F', type = int, default = 16,
                        help = 'number of features in first model')
    parser.add_argument('--activation','-A', default = 'relu', help = 'activation function to use')
    parser.add_argument('--num_convs','-N', type = int, default = 2, help = 'num_convs to use')
    
    args = parser.parse_args()
    
    model = train(**vars(args))



    
    

