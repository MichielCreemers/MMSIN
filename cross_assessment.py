import argparse
import time
import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.MultiModalDataset import MultiModalDataset
from models.main_model import MM_NSSInet


def cross_test_dataset(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_params = np.load(config.minmax_path)
    min_vals, max_vals = scaler_params[0], scaler_params[1]
    
    # required image transformation for model
    transformations_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the dataset
    complete_dataset = MultiModalDataset(projections_dirs=[config.projections_dir], 
                                mos_data_paths=[config.mos_data_path],
                                number_of_projections=config.number_projections,
                                nss_features_dir=[config.nss_path],
                                datasets=config.dataset)
    
    # Apply transformation
    complete_dataset.set_transform(transform=transformations_test)
    
    # Setup DataLoader for testing dataset
    test_loader = DataLoader(complete_dataset, batch_size=config.batch_sizen, shuffle=False, num_workers=0)
    
    # Load the pretrained model
    model = MM_NSSInet()
    model.load_state_dict(torch.load(config.model))
    model = model.to(device)
    model.eval
    predictions = np.zeros(len(complete_dataset))
    actual_scores = np.zeros(len(complete_dataset))
    
    # Do actual testing
    with torch.no_grad():
        for i, (projections, nss, mos) in enumerate(test_loader):
            projections = projections.to(device)
            
            # scale nss features from dataset B to the range of dataset A that is used for training
            nss = (nss - min_vals) / (max_vals - min_vals)
            
            nss = nss.to(device)
            actual_scores[i] = mos.item()
            model_prediction = model(projections, nss)
            predictions[i] = model_prediction.item()
            
            print(f"For the {i}-th point, a MOS score of {predictions[i]:.4f} is predicted compared to the actual MOS score of {actual_scores[i]:.4f}.")
            
    predictions = np.array(predictions)
    actual_scores = np.array(actual_scores)   
    
    # Evaluate model for specific dataset
    srocc = scipy.stats.spearmanr(actual_scores, predictions).correlation
    krocc = scipy.stats.kendalltau(actual_scores, predictions).correlation
    plcc = scipy.stats.pearsonr(actual_scores, predictions)[0]
    rmse = np.sqrt(np.mean((actual_scores - predictions) ** 2))
    
    print(f"Results for dataset {config.dataset}:")
    print(f"SROCC: {srocc:.4f}")
    print(f"KROCC: {krocc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
     
    
# main script
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of the dataset that is used for testing/validation')
    parser.add_argument('--model', type=str, help='path to a pre-trained model')
    parser.add_argument('--projections_dir', type=str, help='path to the directive with the projections for that dataset')
    parser.add_argument('--mos_data_path', type=str, help='path to the csv file with the MOS scores')
    parser.add_argument('nss_path', type=str, help='path to the csv file with the nss features for that dataset')
    parser.add_argument('--batch_size', type=int, help='The batch size for tesing')
    parser.add_argument('--number_projections', type=int, help='The number of projections for each point cloud')
    parser.add_argument('--minmax_path', type=str, help='path to the .npy file with the minmax scaler where the model was trained on')
    
    config = parser.parse_args()
    
    start = time.time()
    cross_test_dataset(config)
    end = time.time()
    print(f"Time taken for cross testing took: {end - start:.4f} seconds.")