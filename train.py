import os
import argparse
import time

import json
import random
import scipy
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from models.main_model import MM_NSSInet
from utils.MultiModalDataset import MultiModalDataset
from utils.loss import L2RankLoss


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

def fit_logistic_model(y_true, y_predict):
    """Fits a logistic model to the given data.

    Args:
        y_true (numpy array): The true output values.
        y_predict (numpy array): The initial prediction values.
    
    Returns:
        numpy array: The output values predicted by the fitted logistic model.
    """
    # Initial parameters guess: upper asymptote, lower asymptote, midpoint x-value, curve steepness
    betas = [np.max(y_true), np.min(y_true), np.mean(y_predict), 0.5]
    
    # Fit the logistic function to the data using curve_fit
    optimized_params, _ = curve_fit(logistic_function, y_predict, y_true, p0=betas, maxfev=100000000)
    
    # Calculate fitted logistic model values
    fitted_values = logistic_function(y_predict, *optimized_params)
    
    return fitted_values

def parse_args():
    """Parse input arguments from JSON config file."""
    with open("config.json", "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    return args


if __name__ == "__main__":
   
    args = parse_args()
    gpu_id = args.gpu
    datasets = args.datasets
    print("Using GPU ID:", gpu_id)
    print("The datasets used are: ", datasets)
    
    