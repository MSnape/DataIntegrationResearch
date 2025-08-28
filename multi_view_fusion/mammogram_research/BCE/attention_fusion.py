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

# Import local code
from mammogram_dataset import MammogramDataset
from resnet_feature_extractor import ResNetFeatureExtractor
from utils import extract_images_by_abnormality_id
from single_view_mammogram import SingleViewMammogram
from classifier_head import ClassifierHead

''' The results from this method were not used in the report as classification accuracy was very poor, 
only a little better than chance. The design is also very similar to the WeightedSumFusionModel which 
outperformed this model'''
class AttentionFusion(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, num_views=2, requires_grad_extractor=False):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(requires_grad=requires_grad_extractor)
        self.num_views = num_views
        self.feature_dim = feature_dim

        # Very simple attention block. Future work will look at using nn.MultiheadAttention
        self.attention_weights_layer = nn.Linear(feature_dim * num_views, num_views)

        # Classifier operates on the weighted sum of features
        self.classifier = ClassifierHead(feature_dim) 

    def forward(self, x_views):
        # Obtain features for each view using the correct feature extractor
        features_per_view = []
        for i in range(self.num_views):
            z_v_i = self.feature_extractor(x_views[i]) # (batch_size, feature_dim)
            features_per_view.append(z_v_i)

        # Stack features to get (batch_size, num_views, feature_dim)
        stacked_features = torch.stack(features_per_view, dim=1) 

        # Flatten for attention weight prediction (batch_size, num_views * feature_dim)
        flat_features = stacked_features.view(stacked_features.size(0), -1)

        # Predict attention weights for each view (batch_size, num_views)
        raw_attention_weights = self.attention_weights_layer(flat_features)
        attention_weights = F.softmax(raw_attention_weights, dim=1) # Ensure weights sum to 1 per sample

        # Apply attention weights: (batch_size, num_views, 1) * (batch_size, num_views, feature_dim)
        # Unsqueeze attention_weights to broadcast correctly
        fused_features = (stacked_features * attention_weights.unsqueeze(-1)).sum(dim=1) # (batch_size, feature_dim)

        output = self.classifier(fused_features)
        return output