# Import all necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import os
import pandas as pd
import pydicom
import re 
import time
import copy
from typing import Literal
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler

# --- MultiViewDataset ---
class MultiViewDataset(Dataset):
    def __init__(self, view_datasets):
        """
        Initializes the MultiViewDataset.

        Args:
            view_datasets (list): A list of Dataset objects, where each
                                  Dataset corresponds to a specific view.
                                  Each dataset in the list must have the
                                  same number of samples and align by index.
                                  e.g., [dataset_view1, dataset_view2, ...]
        """
        if not view_datasets:
            raise ValueError("view_datasets cannot be empty.")
        
        self.view_datasets = view_datasets
        
        # Ensure all view datasets have the same length
        first_len = len(self.view_datasets[0])
        for i, ds in enumerate(self.view_datasets):
            if len(ds) != first_len:
                raise ValueError(
                    f"All view datasets must have the same number of samples. "
                    f"Dataset 0 has {first_len} samples, but dataset {i} has {len(ds)}."
                )
        self._length = first_len # Total number of multi-view samples

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Get data for each view at the given index
        views_data = []
        label = None # Label should be consistent across all views for the same sample

        for i, dataset in enumerate(self.view_datasets):
            view_image, view_label = dataset[idx]
            views_data.append(view_image)
            
            # For multi-view classification, all views of a sample should share the same label.
            # We take the label from the first view and assert consistency.
            if i == 0:
                label = view_label
            else:
                if not torch.equal(label, view_label):
                     print(f"Warning: Labels for index {idx} in view {i} ({view_label}) "
                           f"do not match the label from view 0 ({label}). Using label from view 0.")

        # `views_data` will be a list of tensors, one for each view, e.g., [tensor_view1, tensor_view2]
        # This is what is expected by the LateFusion models in their forward pass.
        return views_data, label
    
    