# Import all necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pandas as pd
import pydicom
import re
import numpy as np
import matplotlib.pyplot as plt 
import time
import copy
from PIL import Image

# Target size for ResNet50
RESNET50_TARGET_IMG_SIZE = 224

# Create the Custom PyTorch Dataset

# --- GrayscaleToResnet50Dataset Class (Single View - Handles All Transforms) ---
class GrayscaleToResnet50Dataset(Dataset):
    def __init__(self, image_arrays, labels, transform=None):
        """
        Initializes the dataset for a single grayscale image view.

        Args:
            image_arrays (list): List of NumPy arrays representing grayscale images.
            labels (list): List of corresponding binary labels.
            transform (callable, optional): A torchvision.transforms.Compose pipeline
                                           that includes all necessary steps:
                                           augmentation (if training), resize, crop,
                                           PIL to Tensor, 1-channel to 3-channel, and normalization.
        """
        self.image_arrays = image_arrays
        self.labels = labels
        self.transform = transform

        # If no transform is provided, use a default that includes base ResNet50 preprocessing
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Convert 1 channel to 3 channels here as part of the pipeline
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                # This normalization is specific for ImageNet which are the weights used for our ResNet-50
                # More information https://paperswithcode.github.io/torchbench/imagenet/ 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
