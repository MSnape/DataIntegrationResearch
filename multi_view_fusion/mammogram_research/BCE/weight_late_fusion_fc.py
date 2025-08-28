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

class WeightedSumFusionModel(nn.Module):
    def __init__(self, num_classes, num_views=2, feature_dim=2048, requires_grad_extractor=False):
        """
        Initializes the WeightedSumFusionModel.

        Args:
            num_classes (int): The number of output classes for classification.
            num_views (int): The number of views to be fused (e.g., 2 for MLO and CC).
            feature_dim (int): The dimension of the feature vectors from the extractor (2048 for ResNet50).
            requires_grad_extractor (bool): Whether the feature extractor's layers should be trainable.
                                           Set to True for fine-tuning, False to keep pre-trained weights frozen.
        """
        super(WeightedSumFusionModel, self).__init__()
        
        self.num_views = num_views
        self.feature_dim = feature_dim

        # Instantiate ResNetFeatureExtractor
        self.feature_extractor = ResNetFeatureExtractor(requires_grad=requires_grad_extractor)

        # Learnable scalar weights for each view
        # We use nn.ParameterList to hold a list of learnable parameters, one for each view.
        self.view_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 / num_views, dtype=torch.float32)) for _ in range(num_views)
        ])

        # Use a Softmax to ensure weights sum to 1 and are positive
        # Softmax will be applied to the stacked weights during the forward pass.
        self.softmax = nn.Softmax(dim=0) 

        # Instantiate ClassifierHead
        self.classifier_head = ClassifierHead(in_features=feature_dim)

    def forward(self, x_views):
        """
        Forward pass for the WeightedSumFusionModel.

        Args:
            x_views (list of torch.Tensor): A list of image tensors, where each tensor
                                            corresponds to a different view (e.g., MLO, CC).
                                            Each tensor should have shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Logits (raw scores) for each class.
        """

        features_per_view = []
        for i in range(self.num_views):
            # Extract features for each view using the shared feature extractor
            z_v_i = self.feature_extractor(x_views[i]) 
            features_per_view.append(z_v_i)

        # Stack the learnable weights and apply softmax to normalize them
        # Creates a 1D tensor of normalized weights (e.g., [w_view1, w_view2, ...])
        normalized_weights = self.softmax(torch.stack(list(self.view_weights)))
        
        # Perform the weighted sum of features
        # Initialize fused_features with zeros of the correct shape
        fused_features = torch.zeros_like(features_per_view[0]) 

        for i in range(self.num_views):
            # Multiply each view's features by its corresponding normalized weight
            fused_features += features_per_view[i] * normalized_weights[i]

        # Pass fused features to classifier head
        logits = self.classifier_head(fused_features)
        return logits