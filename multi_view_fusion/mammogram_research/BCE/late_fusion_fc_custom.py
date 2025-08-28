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

class ClassifierHeadAlt(nn.Module):
    """
    Alternative Classifier Head based on Keras-style architecture.
    """
    def __init__(self, input_dim):
        super(ClassifierHeadAlt, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(128, 1) # Output for binary classification

        # Initialize weights (optional)
        # Deeper Classifier head
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x


class LateFusionCustomClassifier(nn.Module):
    """
    Late Fusion (Fully Connected) model with an alternative Classifier Head.
    """
    def __init__(self, num_classes=1, num_views=2, feature_dim=2048, feature_extractor_requires_grad=False):
        super(LateFusionCustomClassifier, self).__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        # To create the feature vectors we need the ResNet50 feature extractor
        self.feature_extractor = ResNetFeatureExtractor(requires_grad=feature_extractor_requires_grad)

        # View fusion layer (fc)
        self.view_fusion_layer = nn.Sequential(
            nn.Linear(self.feature_dim * self.num_views, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # The classifier part operates on a 1024-dim vector,
        # using the new ClassifierHeadAlt
        self.classifier = ClassifierHeadAlt(1024) # Input dimension is 1024 from view_fusion_layer output 
        # as in Seeland and Mader research

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
        # Pass through the classifer head which has our new ClassifierHeadAlt
        output = self.classifier(fused_features_nn)
        return output