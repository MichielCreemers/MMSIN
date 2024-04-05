import os
import argparse
import time

import random
import scipy
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

def set_random_seed(seed=2024):
    """This function sets the seed for random number generation to ensure reproducibility of the results.
    It affects Python's 'random', 'numpy', and pytorch's random number generators, including CUDA's
    deterministic behavior for operations on the GPU.

    Args:
        seed (int, optional): Value to seed the random number generation. Defaults to 2001.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def logistic_function(X, beta1, beta2, beta3, beta4):
    """This function calculates the logistic function.

    Args:
        X (numpy array): Input data.
        beta1 (float): Upper bounds of the logistic function.
        beta2 (float): Lower bounds of the logistic function.
        beta3 (float): Adjusts the X value at which the curve's midpoint occurs, effectively moving the entire curve along the x-axis.
        beta4 (float): Determines the steepness of the curve.

    Returns:
        yhat
    """
    # Logistic part
    logistic_part = 1 + np.exp(np.negative((np.divide(X - beta3, beta4))))
    # Calculate the predicted values (yhat) usign the logistic function
    y_hat = beta2 + np.divide(beta1 - beta2, logistic_part)
    return y_hat

def fit_function(y_label, y_output):
    """this function fits the input data to the wanted output (MOS)

    Args: 
        y_label (float): input variables 
        y_output (float): the output (MOS) for our input variables to which we want to fit the function

    Returns: 
        y_output_logistic: predicted output values of y_label based off logistic function
    """
    #predict intitial parameters of fit, based on min, max of input val,
    # and mean of output for fit
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]

    #keep optimal parameters (popt), discard covariance matrix (_)
    popt, _ = curve_fit(logistic_function, y_output, \
            y_label, p0=beta, maxfev=100000000)
    
    #evaluate ther iinput variables with optimal parameters
    y_output_logistic = logistic_function(y_output, *popt)

    return y_output_logistic

def parse_args():
    """Parse the input arguments"""
    
    parser = argparse.ArgumentParser(description="first_tests")
    parser.add_argument('--gpu', help="GPU device id to use [0] for dedicated graphics", default=0, type=int)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=30, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=8, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate, how much do want to limit the learning rate')
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--data_dir_2d', default='', type=str, help = 'path to the images')     
    parser.add_argument('--data_dir_nss', default='', type=str, help = 'path to the natural scene statistics')           
    parser.add_argument('--img_length_read', default=4, type=int, help = 'number used immages')
    parser.add_argument('--loss', default='l2rank', type=str)
    parser.add_argument('--database', default='WPC', type=str)
    # parser.add_argument('--k_fold_num', default=5, type=int, help='9 for the SJTU-PCQA, 5 for the WPC, 4 for the WPC2.0')  
    # first tests without k-folds, to keep things a bit easier, just to see wether ourmodel is working

    args = parser.parse_args

    return args

if __name__=='__main__':

    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    #read the inpur para

    print('reading parameters')

    args = parse_args()
    set_random_seed()

    gpu = args.gpu
    cudnn.enabled = True

    #hyperparameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    loss = args.loss

    #input data
    database = args.database
    img_length_read = args.img_length_read
    data_dir_2d = args.data_dir_2d
    data_dir_nss = args.data_dir_nss 

    #model
    model = args.model

    # best_all = np.zeros([args.k_fold_num, 4])

    #check gpu readiness
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    #select the correct database (needs adjustment for k_fold)
    if database == 'SJTU':           
            train_filename_list = 'csvfiles/sjtu_data_info/train.csv'
            test_filename_list = 'csvfiles/sjtu_data_info/test.csv'
    elif database == 'WPC':
        train_filename_list = 'csvfiles/wpc_data_info/train.csv'
        test_filename_list = 'csvfiles/wpc_data_info/test.csv'
    elif database == 'WPC2.0':
        train_filename_list = 'csvfiles/wpc2.0_data_info/train.csv'
        test_filename_list = 'csvfiles/wpc2.0_data_info/test.csv'

    #Crop pictures into a random patch of 224x224 pixels, turn them from rgb value to Tensor (0-255 to 0-1.0) 
    #and finaly normalise them with a value gathered from a large dataset

    transformations_train = transforms.Compose([transforms.RandomCrop(224),transforms.ToTensor(),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])           
    transformations_test = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    





