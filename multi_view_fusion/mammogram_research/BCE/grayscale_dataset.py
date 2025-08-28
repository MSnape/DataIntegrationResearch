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
            
    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.image_arrays)

    def __getitem__(self, idx):
        """
        Loads, transforms, and returns a single grayscale image as a PyTorch Tensor,
        along with its label.
        """
        image_np = self.image_arrays[idx] # This is a NumPy array
        label = self.labels[idx]

        # Convert NumPy array (H, W) to PIL Image ('L' mode for grayscale).
        # Ensure it's 8-bit for PIL Image.fromarray, scale if necessary.
        image_pil = self._normalize_to_uint8(image_np)
        image_pil = Image.fromarray(image_pil, 'L') # 'L' for grayscale

        # Apply the full transformation pipeline (including augmentation, resize, ToTensor, Normalize, 3-channel)
        image_tensor = self.transform(image_pil)
        
        # Ensure label is long tensor for PyTorch cross-entropy loss
        label = torch.tensor(label, dtype=torch.long)

        return image_tensor, label

    def _normalize_to_uint8(self, image_np):
        """Helper to ensure NumPy array is uint8 for PIL.Image.fromarray."""
        if image_np.dtype != np.uint8:
            min_val = image_np.min()
            max_val = image_np.max()
            if max_val == min_val: # Handle constant images to avoid division by zero
                return np.zeros_like(image_np, dtype=np.uint8)
            image_np = (image_np - min_val) / (max_val - min_val) * 255
            image_np = image_np.astype(np.uint8)
        return image_np