import torch
import torch.nn as nn 

class CMA_fusion(nn.Module):
    
    
    
    pass 



class MM_NSSInet(nn.Module):
    
    def __init__(self):
        super(MM_NSSInet, self).__init__()
        self.image_feature_dim = 2048  # the number of feature channels (or dimensions) coming from the 2D image projections after being processed by the image backbone
        self.nss_features_dim = 64  # nss features dimensionality
        self.common_feature_dim = 1024  # dim feature space mapped features before CMA.
        # self.image_backbone --> resnet
        # self.regression --> cma fusion initialiseren met correcte dimensies self.image_backbone
        
        def forward(self, image, nss_features):
            """_summary_

            Args:
                image (Tensor[batch_size, num_projections, C, H, W]): Images for a batch
                nss_features (Tensor[batch_size, 64]): nss features for a batch
            """
            
            # Process 2D projections through image backbone
            # Reshape image from [batch_size, num_projections, C, H, W]
            # to [batch_size * num_projections, C, H, W]
            image = image.view(-1, *image.shape[2:])
            #image_features = self.image_backbone(image)  #Extraction of features
            
            # Flatten output features from the backbone that are in the format 
            # [batch_size * num_projections, image_feature_dim, H', W']
            image_features = torch.flatten(image_features, start_dim=1)
            
            # Re-arrange image_features back to [batch_size, num_projections, image_feature_dim]
            image_features = image_features.view(-1, image.shape[1], self.image_feature_dim)
            
            # Average projection features to get a single feature vecture per image in the batch
            image_features = torch.mean(image_features, dim=1)
            
            # Fuse the features using CMA fusion module and regress to output
            #output = self.regression(image_features, nss_features)
            # return output
            pass
            