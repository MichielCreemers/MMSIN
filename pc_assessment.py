import argparse
import time
import os
import glob
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from utils import projections
from utils.NSS import feature_extract, nss_functions, feature_functions
from models.main_model import MM_NSSInet

def assess_quality(config):

    projections.make_projections(config.pcname,config.projections_folder,config.x_projections, config.y_projections, config.point_size, 'default', False)
    images = glob.glob(f'{config.projections_folder}/*.png')

    transformation = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    
    transformed_imgs = torch.zeros([len(images), 3, 224, 224])

    for i in range(len(images)):
        read_image = Image.open(images[i])
        read_image.convert('RGB')
        read_image = transformation(read_image)
        transformed_imgs[i] = read_image

    

    nss_features = feature_extract.get_feature_vector(config.pcname)

    feature_names = ["l_mean","l_std","l_entropy","a_mean","a_std","a_entropy","b_mean","b_std","b_entropy","curvature_mean","curvature_std","curvature_entropy","curvature_ggd1","curvature_ggd2","curvature_aggd1","curvature_aggd2","curvature_aggd3","curvature_aggd4","curvature_gamma1","curvature_gamma2","anisotropy_mean","anisotropy_std","anisotropy_entropy","anisotropy_ggd1","anisotropy_ggd2","anisotropy_aggd1","anisotropy_aggd2","anisotropy_aggd3","anisotropy_aggd4","anisotropy_gamma1","anisotropy_gamma2","linearity_mean","linearity_std","linearity_entropy","linearity_ggd1","linearity_ggd2","linearity_aggd1","linearity_aggd2","linearity_aggd3","linearity_aggd4","linearity_gamma1","linearity_gamma2","planarity_mean","planarity_std","planarity_entropy","planarity_ggd1","planarity_ggd2","planarity_aggd1","planarity_aggd2","planarity_aggd3","planarity_aggd4","planarity_gamma1","planarity_gamma2","sphericity_mean","sphericity_std","sphericity_entropy","sphericity_ggd1","sphericity_ggd2","sphericity_aggd1","sphericity_aggd2","sphericity_aggd3","sphericity_aggd4","sphericity_gamma1","sphericity_gamma2"]
    
    features_df = pd.DataFrame([nss_features], columns=feature_names)

    scaler = joblib.load('utils/NSS/sc.joblib')

    nss_features = scaler.transform(features_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MM_NSSInet()
    model.load_state_dict(torch.load(config.model))

    model = model.to(device)
    
    model.eval()

    print('Begin inference.')
    with torch.no_grad():
        transformed_imgs = transformed_imgs.to(device).unsqueeze(0)   

        # nss_features_values = nss_features.astype(float).val
        nss_features_tensor = torch.tensor(nss_features, dtype=torch.float).squeeze() 
        nss_features_tensor = nss_features_tensor.to(device).unsqueeze(0)
        outputs = model(transformed_imgs,nss_features_tensor)
        score = outputs.item()
    
    print('Predicted quality score: ' + str(score))

if __name__ == '__main__':

    parser  = argparse.ArgumentParser()

    parser.add_argument('--pcname', type=str, help='path to the point cloud whose quality we want to assess', default='WPC/point_clouds/bag/bag_level_9.ply')
    parser.add_argument('--model', type=str, help='path to the trained model we want to use to assess the point cloud', default="ckpts/WPC_300_epoch/['WPC']_0_best_model.pth")
    parser.add_argument('--x_projections', type=int, help='the number of projections along the x-axis to take', default=4)
    parser.add_argument('--y_projections', type=int, help='the number of projections along the y-axis to take', default=4)
    parser.add_argument('--point_size', type=str, help='size of the projections', default=2)
    parser.add_argument('--projections_folder', type=str, help='path to folder to save images', default='test')
    
    config = parser.parse_args()

    start = time.time()

    assess_quality(config)

    
    end = time.time()

    print(f"assessment took ", end-start ," seconds")
