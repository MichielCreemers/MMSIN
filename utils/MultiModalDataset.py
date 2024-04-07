import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image


class MultiModalDataset(Dataset):
    
    """
    NJAAA
    """
    
    def __init__(self, projections_dirs: list[str], nss_features_dir: list[str], pcl_dirs: list[str], mos_data_paths: list[str], number_of_projections, datasets=1, model="nss1", crop_size=400, train=True):
        
        """
        NJAAA
        
        Args:
        * projections_dir (list[str]):  
        * number_of_projections (int): The number of 2D images to be read/processed for each instance.
        * dataset (int, optional): Either 1, 2 or 3 depending on `HOW MANY` datasets you want to train.
        * train (bool, optional): A flag to indicate whether the dataset is used for training.
        """
        
        super(MultiModalDataset, self).__init__()
        
        # Validation received parameters
        if datasets not in [1, 2, 3]:
            raise ValueError(f"Invalid amount of datasets: {datasets}. Expects either '1', '2' or '3'.")
        
        if (len(projections_dirs) != datasets) or (len(mos_data_paths) != datasets):
            raise ValueError("Make sure the correct paths for each dataset are provided.")
        
        if model not in ["nss1"]:
            raise ValueError(f"Invalid model: {model}, please select a valid model: nss1, ... or ...")

        # If multiple datasets are used, their MOS scores should be concatenated into one dataframe
        mos_data_frames = [pd.read_csv(path, sep=',', header=0) for path in mos_data_paths[:datasets]]
        mos_data = pd.concat(mos_data_frames, ignore_index=True)
        self.pointcloud_names = mos_data[['name']]  # [['']] keeps DataFrame properties ~keeps column name
        self.pointcloud_mos = mos_data['mos']
        
        # If model nss1 is used --> also load the features
        if model=="nss1":
            
            # Quick check
            if len(nss_features_dir) != datasets:
                raise ValueError(f"Expected number of nss feature csv's: {datasets}, only {len(nss_features)} were given.")
            
            nss_features_frames = [pd.read_csv(path, sep=',', header=0) for path in nss_features_dir[:datasets]]
            nss_features = pd.concat(nss_features_frames, ignore_index=True)
            self.nss_features_names = nss_features[['name']]
            
        self.number_of_projections = number_of_projections
        self.crop_size = crop_size
        self.train = train
        
        
        self.length = len(self.pointcloud_names)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        pass
        
        
        
        
        
        
        
        