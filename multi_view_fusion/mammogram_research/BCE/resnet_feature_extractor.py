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


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet Feature Extractor i.e Resnet Architecture but without the final Fully Connected layer
    """
    def __init__(self, requires_grad=False):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Remove the final Fully Connected layer
        
        # Freeze or unfreeze the pre-trained layers , this is for future work, we do not unfreeze earlier
        # layers currently but this would be looked at in the future 
        for param in self.features.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        # x is expected to be a batch of images (batch_size, 3, H, W)
        z_v = self.features(x)
        # Flatten the output of avgpool (batch_size, 2048, 1, 1) to (batch_size, 2048)
        z_v = torch.flatten(z_v, 1) 
        return z_v


