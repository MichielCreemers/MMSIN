import torch
import torch.nn as nn 
from models.backbones import resnet50
from models.transformer import TransformerEncoderLayer_CMA

class CMA_fusion(nn.Module):
    
    def __init__(self, image_inplanes, nss_inplanes, cma_planes=1024):
        
        super(CMA_fusion, self).__init__()
        
        self.encoder = TransformerEncoderLayer_CMA(d_model = cma_planes, nhead = 8, dim_feedforward = 2048, dropout = 0.1)  
        self.linear1 = nn.Linear(image_inplanes, cma_planes)  # 2048 -> 1024
        self.linear2 = nn.Linear(nss_inplanes, cma_planes)  # 64 -> 1024
        self.quality1 = nn.Linear(cma_planes * 4, cma_planes * 2)
        self.quality2 = nn.Linear(cma_planes * 2, 1)
        self.image_bn = nn.BatchNorm1d(cma_planes)
        self.nss_bn = nn.BatchNorm1d(cma_planes)
        
    def forward(self, image_features, nss_features):
        
        #  map the image and NSS features from their original dimensionality 
        # (image_inplanes and nss_inplanes) to the common feature space dimensionality (cma_planes).
        image_features = self.linear1(image_features)
        # print("image_features before bn: ", image_features.shape)
        # if image_features.size(0) > 1:
        image_features = self.image_bn(image_features)
        # print("image_features after bn: ", image_features.shape)
        nss_features = self.linear2(nss_features)
        # print("nss_features shape: ", nss_features.shape)
        # if nss_features.size(0) > 1:
        nss_features = self.nss_bn(nss_features)
        
        # Add singleton dimension -> MultiheadAttention requires [seq_len, batch_size, feature_size] because batch_first False by default in MultiHeadAttention !!!
        image_features = image_features.unsqueeze(0)  # [batch_size, 1024] -> [1, batch_size, 1024]
        nss_features = nss_features.unsqueeze(0)  # ""
        
        # cma
        img_a, nss_a = self.encoder(image_features, nss_features)
        
        # Concatenate original features and attention-augmented features to enrich feature representation
        output = torch.cat((image_features, img_a, nss_a, nss_features), dim=2)
        
        # remove extra dimension introduced by unqsueeze
        output = output.squeeze(0)
        
        # Reduce dimensionality to a single quality score
        output = self.quality1(output)
        output = self.quality2(output)
        
        return output
        
    
    
    


class MM_NSSInet(nn.Module):
    
    def __init__(self):
        super(MM_NSSInet, self).__init__()
        self.image_feature_dim = 2048  # the number of feature channels (or dimensions) coming from the 2D image projections after being processed by the image backbone
        self.nss_features_dim = 64  # nss features dimensionality
        self.common_feature_dim = 1024  # dim feature space mapped features before CMA.
        self.image_backbone = resnet50(pretrained=True)
        self.regression = CMA_fusion(image_inplanes=self.image_feature_dim, 
                                     nss_inplanes=self.nss_features_dim, 
                                     cma_planes=self.common_feature_dim)
        
    def forward(self, image, nss_features):
        """_summary_

        Args:
            image (Tensor[batch_size, num_projections, C, H, W]): Images for a batch
            nss_features (Tensor[batch_size, 64]): nss features for a batch
        """
        image_shape = image.shape
        # Process 2D projections through image backbone
        # Reshape image from [batch_size, num_projections, C, H, W]
        # to [batch_size * num_projections, C, H, W]
        image = image.view(-1, *image.shape[2:])
        # print(image.shape)
        image_features = self.image_backbone(image)  #Extraction of features
        # print("image features: ", image_features.shape)
        # Flatten output features from the backbone that are in the format 
        # [batch_size * num_projections, image_feature_dim, H', W']
        image_features = torch.flatten(image_features, start_dim=1)
        # print("image features: ", image_features.shape)
        # Re-arrange image_features back to [batch_size, num_projections, image_feature_dim]
        image_features = image_features.view(-1, image_shape[1], self.image_feature_dim)
        # print("image features (bs, np, 2048): ", image_features.shape)
        # Average projection features to get a single feature vecture per image in the batch
        image_features = torch.mean(image_features, dim=1)
        # print("image features: ", image_features.shape)
        # Fuse the features using CMA fusion module and regress to output
        output = self.regression(image_features, nss_features)
        # print("Output shape: ", output.shape)
        return output
        
            