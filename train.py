import os
import argparse
import time

import json
import random
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold, train_test_split
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

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
    print('*****************************************************************************')
    args = parse_args()
    set_random_seed()
    cudnn.enabled = True
    
    # Hyperparameters
    gpu = args.gpu
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    k_fold_num = args.k_fold_num
    
    # Input data
    projections_dirs = args.projections_dirs
    mos_data_paths = args.mos_data_paths
    number_of_projections = args.number_of_projections
    nss_features_paths = args.nss_features_paths
    datasets = args.datasets
    
    print("The dataset(s) used is/are: ", datasets)
    
    # GPU readiness
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print("Using GPU")
    
    # See https://pytorch.org/hub/pytorch_vision_resnet/
    transformations_train = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformations_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print('*****************************************************************************')
    # Load the MultiModalDataset
    complete_dataset = MultiModalDataset(projections_dirs=projections_dirs, 
                                mos_data_paths=mos_data_paths,
                                number_of_projections=number_of_projections,
                                nss_features_dir=nss_features_paths,
                                datasets=datasets)
    
    # Split the dataset into training and tens sets (80% train & 20% test)
    train_indices, test_indices = train_test_split(range(len(complete_dataset)), test_size=0.2, random_state=42)
    
    # Create subset for training and testing
    train_dataset = Subset(complete_dataset, train_indices)
    test_dataset = Subset(complete_dataset, test_indices)
    
    # Start kfold cross validation loop
    kf = KFold(n_splits=k_fold_num, shuffle=True, random_state=42)
    for fold, (train_ids, val_ids) in enumerate(kf.split(range(len(complete_dataset)))):
        print(f"Starting fold {fold+1}/{k_fold_num}")
        
        # Subset training dataset in training and vailidation
        train_subset = Subset(train_dataset, train_ids)
        val_subset = Subset(train_dataset, val_ids)
        
        # Initialize data loaders for current fold
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        # Initialize model
        if args.model == "nss1":
            model = MM_NSSInet()
            model = model.to(device)
        
        if args.loss = "l2rank"
