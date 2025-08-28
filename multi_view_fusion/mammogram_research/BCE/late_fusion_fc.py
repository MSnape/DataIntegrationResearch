import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import random
import numpy as np

# Important local code
from mammogram_dataset import MammogramDataset
from resnet_feature_extractor import ResNetFeatureExtractor
from utils import extract_images_by_abnormality_id
from single_view_mammogram import SingleViewMammogram
from classifier_head import ClassifierHead


class LateFusionFC(nn.Module):
    """
    Late Fusion (Fully Connected) model from Seeland and Mader paper 
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245230
    """
    def __init__(self, num_classes, num_views=2, feature_dim=2048, feature_extractor_requires_grad=False):
        super(LateFusionFC, self).__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        # To create the feature vectors we need the ResNet50 feature extractor
        self.feature_extractor = ResNetFeatureExtractor(requires_grad=feature_extractor_requires_grad)
        
        # View fusion layer (fully connected (fc))
        self.view_fusion_layer = nn.Sequential(
            nn.Linear(self.feature_dim * self.num_views, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # The classifier part operates on a 1024-dim vector
        self.classifier = ClassifierHead(1024)
        
        # Initialize weights for new layers
        nn.init.kaiming_normal_(self.view_fusion_layer[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.view_fusion_layer[0].bias)

    def forward(self, x_views):
        
        # x_views is a list of nV image tensors, each (batch_size, 3, H, W)
        features_per_view = []
        for i in range(self.num_views):
            z_v_i = self.feature_extractor(x_views[i]) # (batch_size, D)
            features_per_view.append(z_v_i)

        # Concatenate feature vectors (batch_size, D * nV)
        fused_features_concatenated = torch.cat(features_per_view, dim=1)

        # Pass through the view fusion layer
        fused_features_nn = self.view_fusion_layer(fused_features_concatenated)
        
        output = self.classifier(fused_features_nn)
        return output
    
   