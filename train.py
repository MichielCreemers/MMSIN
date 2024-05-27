import os
import argparse
import time

import json
import random
import copy
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


def update_transform(dataset, transform):
    dataset.set_transform(transform)

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

    best_all = np.zeros([k_fold_num, 4])

    print("The dataset(s) used is/are: ", datasets)
    
    # GPU readiness
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    #     print("Using GPU")
    
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
    # train_indices, test_indices = train_test_split(range(len(complete_dataset)), test_size=0.0, random_state=42)
    
    # train_dataset = Subset(complete_dataset, train_indices)
    # test_dataset = Subset(complete_dataset, test_indices)
    
    # Start kfold cross validation loop
    kf = KFold(n_splits=k_fold_num, shuffle=True, random_state=42)
    for fold, (train_ids, val_ids) in enumerate(kf.split(range(len(complete_dataset)))):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print("Using GPU")
        
        print(f"Starting fold {fold+1}/{k_fold_num}")
        
        # Create copies
        train_dataset_clone = copy.deepcopy(complete_dataset)
        val_dataset_clone = copy.deepcopy(complete_dataset)
        
        train_dataset_clone.set_transform(transform=transformations_train)
        val_dataset_clone.set_transform(transform=transformations_test)
        
        train_subset = Subset(train_dataset_clone, train_ids)
        val_subset = Subset(val_dataset_clone, val_ids)
        
        # Initialize data loaders for current fold ---- IN OUT LOOP?????? __________!_!_____________!_!_!_______________!_!____________
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize model, criterion, optimizer
        if args.model == "nss1":
            model = MM_NSSInet()
            model = model.to(device)
        
        if args.loss == "l2rank":
            criterion = L2RankLoss().to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_rate)
        print(f"Using Adam optimize with initial learning rate of {learning_rate}")
        #let the optimizer adjust its learning rate every 8 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=8, gamma=0.9)
        
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print("starting the training")
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        best_test_criterion = -1 
        overall_best_test_criterion = -1
        best = np.zeros(4)
        epoch_split=0

        for epoch in range(num_epochs):
            if epoch % 100 == 0:
                best_test_criterion = -1
                epoch_split += 1
            n_train = len(train_subset)
            n_val  = len(val_subset)

            model.train()

            start = time.time()
            batch_losses = []
            batch_losses_each_disp = []
            
            x_output = np.zeros(n_train)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!
            x_val  = np.zeros(n_train)

            for i, (imgs, nss, mos) in enumerate(train_loader):
                imgs = imgs.to(device)
                # print("images:", imgs.shape)
                # nss  = nss[:,np.newaxis]
                nss  = nss.to(device)
                # print("nss:", nss.shape)

                #!!!!!!!!
                mos  = mos[:,np.newaxis]
                #!!!!!!!!
                mos = mos.to(device)

                mos_output = model(imgs, nss)

                loss = criterion(mos_output, mos)
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())

                optimizer.zero_grad()
                torch.autograd.backward(loss)
                optimizer.step()

            # print("Images on device:", imgs.device)
            # print("Model on device:", next(model.parameters()).device)
            
            avg_loss = sum(batch_losses) / (len(train_subset) // batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            scheduler.step()
            lr_current = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr_current[0]))

            end = time.time()
            print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end-start))

            model.eval()
            y_output = np.zeros(n_val)
            y_val   = np.zeros(n_val)

            with torch.no_grad():
                for i, (imgs, nss, mos) in enumerate(val_loader):
                    imgs = imgs.to(device)
                    # nss = nss[:, np.newaxis]
                    nss = nss.to(device)
                    # print("The MOS shape: ", mos.shape)
                    y_val[i] = mos.item()
                    outputs = model(imgs, nss)
                    y_output[i] = outputs.item()

                y_output_logistic = fit_logistic_model(y_val, y_output)
                test_PLCC = stats.pearsonr(y_output_logistic, y_val)[0]
                test_SROCC = stats.spearmanr(y_output, y_val)[0]
                test_RMSE = np.sqrt(((y_output_logistic-y_val) ** 2).mean())
                test_KROCC = scipy.stats.kendalltau(y_output, y_val)[0]
                print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))

                if test_SROCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    torch.save(model.state_dict(), 'ckpts/' + str(datasets) + '_' + str(fold) + '_' + str(epoch_split)+'_best_model.pth')
                    # scio.savemat(trained_model_file+'.mat',{'y_pred':y_pred,'y_test':y_test})
                    best_test_criterion = test_SROCC  # update best val SROCC
                    if test_SROCC > overall_best_test_criterion:
                        best[0:4] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
                        overall_best_test_criterion = test_SROCC

                    print("Update the best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))
        print(datasets)
        best_all[fold-1, :] = best
        print("The best Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1], best[2], best[3]))
        print('*************************************************************************************************************************')
    
    # average score
    best_mean = np.mean(best_all, 0)
    print('*************************************************************************************************************************')
    print("The mean val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_mean[0], best_mean[1], best_mean[2], best_mean[3]))
    print('*************************************************************************************************************************')


