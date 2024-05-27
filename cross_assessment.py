import argparse
import time
import numpy as np
import scipy.stats
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.MultiModalDataset import MultiModalDataset
from models.main_model import MM_NSSInet
from sklearn.preprocessing import MinMaxScaler


def cross_test_dataset(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_params = np.load(config.minmax_path)
    min_vals, scale = scaler_params[0], scaler_params[1]
    
    # required image transformation for model
    transformations_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    start_time = time.time()
    # Load the dataset
    complete_dataset = MultiModalDataset(projections_dirs=[config.projections_dir], 
                                mos_data_paths=[config.mos_data_path],
                                number_of_projections=config.number_projections,
                                nss_features_dir=[config.nss_path],
                                datasets=[config.dataset])
    
    # Apply transformation
    complete_dataset.set_transform(transform=transformations_test)
    
    # Setup DataLoader for testing dataset
    test_loader = DataLoader(complete_dataset, batch_size=1, drop_last=True, shuffle=False, num_workers=0)
    end_time = time.time()
    duration = end_time-start_time
    print("preprocessing took: " + str(duration) + "s")

    start_time = time.time()
    # Load the pretrained model
    model = MM_NSSInet()
    model.load_state_dict(torch.load(config.model))
    model = model.to(device)
    model.eval()
    predictions = np.zeros(len(complete_dataset))
    actual_scores = np.zeros(len(complete_dataset))
    scaler = MinMaxScaler()
    scaler.min_ = min_vals
    scaler.scale_ = scale
    end_time = time.time()
    duration = end_time-start_time
    print("loading the model took: " + str(duration) + "s")
    # Do actual testing
    df_timings = pd.DataFrame(columns=['time'])

    with torch.no_grad():
        for i, (projections, nss, mos) in enumerate(test_loader):
            start_time = time.time()

            projections = projections.to(device)
            
            # scale nss features from dataset B to the range of dataset A that is used for training
            nss_scaled = scaler.transform(nss)
            nss_scaled = torch.tensor(nss_scaled, dtype=torch.float).squeeze()
            nss_scaled = nss_scaled.to(device).unsqueeze(0)
            print(nss_scaled.shape)
            actual_scores[i] = mos.item()
            model_prediction = model(projections, nss_scaled)
            predictions[i] = model_prediction.item()
            
            print(f"For the {i}-th point, a MOS score of {predictions[i]:.4f} is predicted compared to the actual MOS score of {actual_scores[i]:.4f}.")
            end_time = time.time()
            duration = end_time-start_time
            df_timings.loc[len(df_timings)] = duration

    mean  = df_timings['time'].mean()
    stdev = df_timings['time'].std()
    min_time = df_timings['time'].min()
    max_time = df_timings['time'].max()

    print(f"Average time: {mean}")
    print(f"Standard deviation: {stdev}")
    print(f"Minimum time: {min_time}")
    print(f"Maximum time: {max_time}")


    predictions = np.array(predictions)
    actual_scores = np.array(actual_scores)   
    
    # Evaluate model for specific dataset
    srocc = scipy.stats.spearmanr(actual_scores, predictions).correlation
    krocc = scipy.stats.kendalltau(actual_scores, predictions).correlation
    plcc = scipy.stats.pearsonr(actual_scores, predictions)[0]
    rmse = np.sqrt(np.mean((actual_scores - predictions) ** 2))
    
    print(f"Results for dataset {config.dataset} tested on model {config.model}:")
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
    parser.add_argument('--nss_path', type=str, help='path to the csv file with the nss features for that dataset')
    parser.add_argument('--batch_size', type=int, help='The batch size for tesing')
    parser.add_argument('--number_projections', type=int, help='The number of projections for each point cloud')
    parser.add_argument('--minmax_path', type=str, help='path to the .npy file with the minmax scaler where the model was trained on')
    
    config = parser.parse_args()
    
    start = time.time()
    cross_test_dataset(config)
    end = time.time()
    print(f"Time taken for cross testing took: {end - start:.4f} seconds.")