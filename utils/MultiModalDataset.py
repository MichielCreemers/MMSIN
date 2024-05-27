import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data
from PIL import Image


class MultiModalDataset(Dataset):
    
    """
    NJAAA
    """
    
    def __init__(self, projections_dirs: list[str], mos_data_paths: list[str], number_of_projections, nss_features_dir=[], pcl_dirs=[], datasets=["sjtu"], model="nss1", crop_size=224):
        
        """
        NJAAA
        
        Args:
        * 
        """
        
        super(MultiModalDataset, self).__init__()
        
        # Validation received parameters
        self.number_of_datasets = len(datasets)
        
        if (len(projections_dirs) != self.number_of_datasets):
            raise ValueError(f"Make sure the same amount of dirs are given, projection dirs: {len(projections_dirs)}, expected: {self.number_of_datasets}!")
        
        if (len(mos_data_paths) != self.number_of_datasets):
            raise ValueError(f"For using {self.number_of_datasets} datasets, {self.number_of_datasets} mos.csv's are expected. {len(mos_data_paths)} were given!")
            
        if model not in ["nss1"]:
            raise ValueError(f"Invalid model: {model}, please select a valid model: nss1, ... or ...")

        
        
        # If multiple datasets are used, their MOS scores should be concatenated into one dataframe
        mos_data_frames = [pd.read_csv(path, sep=',', header=0) for path in mos_data_paths[:self.number_of_datasets]]
        mos_data = pd.concat(mos_data_frames, ignore_index=True)
        self.pointcloud_names = mos_data[['name']]  # [['']] keeps DataFrame properties ~keeps column name
        self.pointcloud_mos = mos_data['mos']
        self.__ds = mos_data['dataset']  # Easier to know which dataset a datapoint is from later on
        self.dataset_lengths = mos_data['dataset'].value_counts().to_dict()
        
        # If model nss1 is used --> load the features
        if model=="nss1":
            
            # Quick check
            if len(nss_features_dir) != self.number_of_datasets:
                raise ValueError(f"Expected number of nss feature csv's: {self.number_of_datasets}, only {len(nss_features_dir)} were given.")
            
            nss_features_frames = [pd.read_csv(path, sep=',', header=0) for path in nss_features_dir[:self.number_of_datasets]]
            self.nss_features = pd.concat(nss_features_frames, ignore_index=True)
            #self.nss_features_names = nss_features[['name']]
        
        self.projections_dirs = projections_dirs
        self.datasets = datasets    
        self.number_of_projections = number_of_projections
        self.transform = None
        self.crop_size = crop_size  # Standard resnet50, ... etc expects at minimum 224x224 inputs 
        self.length = len(self.pointcloud_names)
    
    def set_transform(self, transform):
        self.transform = transform
        
    def __len__(self):
        return self.length
    
    def _check_dataset(self, index: int):
        """Returns the name of the dataset a given datapoint belongs to.
        --> Make sure the list of dirs used when making the class are all relative to eachother.

        Args:
            index (int): The index of datapoint in the concatenated dataset.
            
        Returns:
            (str, int): (The name of the dataset the datapoint belongs to, index in dir)
        """
        ds = self.__ds.iloc[index]
        ds_idx = self.datasets.index(ds)
        return (ds, ds_idx)
        
    
    
    def __getitem__(self, index):
        
        # First check what dataset the datapoint with the corresponding index is from
        dataset, ds_idx = self._check_dataset(index)
        
        # Projections for current index
        image_name = self.pointcloud_names.iloc[index, 0]
        projections_dir = os.path.join(self.projections_dirs[ds_idx], image_name)
        
        # Initialize a tensor to store the transformed images
        transformed_images = torch.zeros([self.number_of_projections, 3, self.crop_size, self.crop_size])
        
        # Load and transform the images
        for i in range(self.number_of_projections):
            image_path = os.path.join(projections_dir, f"projection_{i}.png")
            try:
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                transformed_images[i] = image
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
        
        # NSS features for the current index, the name should match that of the image
        nss_features_row = self.nss_features.iloc[index]
        # nss_features_row = self.nss_features[self.nss_features['name'] == image_name]
        
        # 'name' column is not a feature
        nss_features_values = nss_features_row.drop('name').astype(float).values
        nss_features_tensor = torch.tensor(nss_features_values, dtype=torch.float).squeeze() 
        
        # Fetch the MOS value       
        y_mos = torch.tensor([self.pointcloud_mos.iloc[index]], dtype=torch.float)
        
        return transformed_images, nss_features_tensor, y_mos
            
    
            
       
        
        
        
        
        
        