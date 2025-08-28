# Import all necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pandas as pd
import pydicom
import re
import numpy as np
import time
import copy
from typing import Literal
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold

# Important local code
from mammogram_dataset import MammogramDataset
from grayscale_dataset import GrayscaleToResnet50Dataset
from utils import extract_images_by_abnormality_id, show_confusion_matrix, print_performance_metrics, get_performance_metrics
from nlm_transform import NLMTransform

class SingleViewMammogram:
    """
    Single View Model which stores either CC or MLO images for all the ROI entities.
    """
    def __init__(self, train_dataset: MammogramDataset, test_dataset: MammogramDataset, view_type: Literal["CC", "MLO"]):
        """
        Initializes the dataset.
        Args:
            train_dataset (MammogramDataset): Training dataset.
            test_dataset (MammogramDataset): Test dataset.
            view_type (Literal): Name of the view type, used for various print statements.
        """
        self.view_type = view_type
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.saved_model_path = ""

        self.train_unique_image_ids = [entity['patient_id']+entity['breast_side']+ str(entity['abnormality id']) for entity in train_dataset
        if len(entity['images'])==4]
        self.dataset_train_view = extract_images_by_abnormality_id(train_dataset, view_type)
    
        self.dataset_train_breast_density = [entity['breast density'] for entity in train_dataset
        if len(entity['images'])==4]
    
        self.dataset_train_pathology_labels = [entity['pathology'] for entity in train_dataset
        if len(entity['images'])==4]

        self.dataset_train_pathology_binary_labels = [   0 if "BENIGN" in label.upper() else  1 if "MALIGNANT" in label.upper() else -1  # Default for labels that are neither BENIGN nor MALIGNANT
        for label in self.dataset_train_pathology_labels]
        if -1 in self.dataset_train_pathology_binary_labels:
            raise ValueError("One or more labels could not be categorized (resulted in -1) in training data.")

        # Create test views
        self.test_unique_image_id = [entity['patient_id']+entity['breast_side']+ str(entity['abnormality id']) for entity in self.test_dataset if len(entity['images'])==4]
        self.dataset_test_view = extract_images_by_abnormality_id(self.test_dataset, self.view_type)
    
        self.dataset_test_breast_density = [entity['breast density'] for entity in self.test_dataset if len(entity['images'])==4]
        self.dataset_test_pathology_labels = [entity['pathology'] for entity in self.test_dataset
        if len(entity['images'])==4]
        # This does set BENIGN_WITH_CALLBACK to BENIGN
        self.dataset_test_pathology_binary_labels = [   0 if "BENIGN" in label.upper() else  1 if "MALIGNANT" in label.upper() else -1  # Default for labels that are neither BENIGN nor MALIGNANT
            for label in self.dataset_test_pathology_labels]

        if -1 in self.dataset_test_pathology_binary_labels:
            raise ValueError("One or more labels could not be categorized (resulted in -1).")
     
    def set_train_val_datasets(self, train_dataset:GrayscaleToResnet50Dataset,val_dataset:GrayscaleToResnet50Dataset):
        self.torch_dataset_train_view =train_dataset
        self.torch_dataset_validation_view = val_dataset
    
    def set_test_view(self, test_view:GrayscaleToResnet50Dataset):
        self.torch_dataset_test_view =test_view

    def CreateResnet50Dataset(self, seed=42):
        """
        Create the pytorch datasets, one for each view
            
        """
        VAL_TOTAL_RATIO = 0.15
        total_images_count = len(self.dataset_train_view) + len(self.dataset_test_view)
        val_size = int(VAL_TOTAL_RATIO * total_images_count)
        train_size = len(self.dataset_train_view) - val_size 

        train_and_val_dataset = GrayscaleToResnet50Dataset(self.dataset_train_view, self.dataset_train_pathology_binary_labels)
     
        self.torch_dataset_test_view = GrayscaleToResnet50Dataset(self.dataset_test_view, self.dataset_test_pathology_binary_labels)

        self.torch_dataset_train_view, self.torch_dataset_validation_view = random_split(
            train_and_val_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed) # For reproducibility
            )
            
    def CreateResnet50DatasetAugment(self):
        """
        Create the PyTorch datasets for a single view, with specific augmentations
        (horizontal mirroring) added to the training set *after* the initial split.
        """
        VAL_TOTAL_RATIO = 0.15
        total_images_count = len(self.dataset_train_view) + len(self.dataset_test_view)
        val_size = int(VAL_TOTAL_RATIO * total_images_count)
        train_size = len(self.dataset_train_view) - val_size 
        
        # Define transformation pipeline for training data (includes base + augmentations)
        base_train_eval_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0,1])
            # Convert 1 channel to 3 channels here as part of the pipeline
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

        # Define transformation for horizontal mirroring
        horizontal_flip_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0), # p=1.0 ensures every image is flipped
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
     
        train_and_val_dataset = GrayscaleToResnet50Dataset(self.dataset_train_view, self.dataset_train_pathology_binary_labels)
     
        # 2. Perform the random split on the indices of the original data.
        # This gives us the indices for the training and validation portions.
        train_indices, val_indices = random_split(
            range(len(train_and_val_dataset)), # Split indices, not the dataset itself
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42) # For reproducibility
        )
        
        # Extract actual indices from Subset objects
        train_indices = train_indices.indices
        val_indices = val_indices.indices

        # 3. Create the original training dataset subset (no augmentations, just base preprocessing)
        self.torch_dataset_train_view_original = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in train_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in train_indices],
            transform=base_train_eval_transforms # Apply standard transforms
        )

        # 4. Create augmented training dataset subset for horizontal mirroring
        self.torch_dataset_train_view_flipped = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in train_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in train_indices],
            transform=horizontal_flip_transform # Apply horizontal flip transform
        )

        # 5. Concatenate all training datasets
        # This creates a training set that is 2x the size of the original training split (original + flipped).
        self.torch_dataset_train_view = ConcatDataset([
            self.torch_dataset_train_view_original,
            self.torch_dataset_train_view_flipped
        ])

        # 6. Create the validation dataset (no augmentations)
        self.torch_dataset_validation_view = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in val_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in val_indices],
            transform=base_train_eval_transforms # Apply standard transforms
        )
        
        # 7. Create the test dataset (no augmentations)
        self.torch_dataset_test_view = GrayscaleToResnet50Dataset(
            self.dataset_test_view, 
            self.dataset_test_pathology_binary_labels,
            transform=base_train_eval_transforms # Apply standard transforms
        )
        
        print(f"Original training split size: {len(self.torch_dataset_train_view_original)}")
        print(f"Horizontally flipped augmented training split size: {len(self.torch_dataset_train_view_flipped)}")
        print(f"Total training dataset size (original + augmented): {len(self.torch_dataset_train_view)}")
        print(f"Validation dataset size: {len(self.torch_dataset_validation_view)}")
        print(f"Test dataset size: {len(self.torch_dataset_test_view)}")

    def CreateResnet50DatasetAugmentDenoise(self):
        """
        Create the PyTorch datasets for a single view, with specific augmentations
        (horizontal mirroring) added to the training set *after* the initial split.
        Includes NLM denoising.
        """
        VAL_TOTAL_RATIO = 0.15
        total_images_count = len(self.dataset_train_view) + len(self.dataset_test_view)
        val_size = int(VAL_TOTAL_RATIO * total_images_count)
        train_size = len(self.dataset_train_view) - val_size 
        
        # --- NLM Denoising Parameters ---
        # Adjust 'h' based on the noise level in mammograms.
        # For medical images, we need to balance noise reduction with detail preservation.
        nlm_h_param = 0.05 # Small h to test, a higher 'h' means more aggressive denoising but also more blurring.
        nlm_patch_size = 7 # Common patch size
        nlm_patch_distance = 15 # Common search window size

        # Define NLM transform instance
        nlm_transform_instance = NLMTransform(
            h=nlm_h_param,
            patch_size=nlm_patch_size,
            patch_distance=nlm_patch_distance,
            multichannel=False # Grayscale mammograms
        )

        # Define transformation pipeline for training data (includes base + augmentations)
        base_train_eval_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # Add NLM denoising here, after cropping but before converting to Tensor
            nlm_transform_instance,
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0,1])
            # Convert 1 channel to 3 channels 
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

        # Define transformation for horizontal mirroring
        # Apply NLM to the flipped image for denoised post-flip
        horizontal_flip_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0), # p=1.0 ensures every image is flipped
            # Add NLM denoising for flipped images
            nlm_transform_instance, 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
     
        train_and_val_dataset = GrayscaleToResnet50Dataset(self.dataset_train_view, self.dataset_train_pathology_binary_labels)
     
        # Perform the random split on the indices of the original data.
        # This gives us the indices for the training and validation portions.
        train_indices, val_indices = random_split(
            range(len(train_and_val_dataset)), # Split indices, not the dataset itself
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42) # For reproducibility
        )
        
        # Extract actual indices from Subset objects
        train_indices = train_indices.indices
        val_indices = val_indices.indices

        # Create the original training dataset subset (no augmentations)
        self.torch_dataset_train_view_original = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in train_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in train_indices],
            transform=base_train_eval_transforms # Apply standard transforms including NLM
        )

        # Create augmented training dataset subset for horizontal mirroring
        self.torch_dataset_train_view_flipped = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in train_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in train_indices],
            transform=horizontal_flip_transform # Apply horizontal flip transform including NLM
        )

        # Concatenate all training datasets
        # This creates a training set that is 2x the size of the original training split (original + flipped).
        self.torch_dataset_train_view = ConcatDataset([
            self.torch_dataset_train_view_original,
            self.torch_dataset_train_view_flipped
        ])

        # Create the validation dataset (no augmentations)
        self.torch_dataset_validation_view = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in val_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in val_indices],
            transform=base_train_eval_transforms # Apply standard transforms including NLM
        )
        
        # Create the test dataset (no augmentations)
        self.torch_dataset_test_view = GrayscaleToResnet50Dataset(
            self.dataset_test_view, 
            self.dataset_test_pathology_binary_labels,
            transform=base_train_eval_transforms # Apply standard transforms including NLM
        )
        
        print(f"Original training split size: {len(self.torch_dataset_train_view_original)}")
        print(f"Horizontally flipped augmented training split size: {len(self.torch_dataset_train_view_flipped)}")
        print(f"Total training dataset size (original + augmented): {len(self.torch_dataset_train_view)}")
        print(f"Validation dataset size: {len(self.torch_dataset_validation_view)}")
        print(f"Test dataset size: {len(self.torch_dataset_test_view)}")

    def CreateResnet50DatasetDenoise(self):
        """
        Create the PyTorch datasets for a single view, with only NLM denoising
        applied to the training, validation, and test sets. No augmentations.
        """
        VAL_TOTAL_RATIO = 0.15
        total_images_count = len(self.dataset_train_view) + len(self.dataset_test_view)
        val_size = int(VAL_TOTAL_RATIO * total_images_count)
        train_size = len(self.dataset_train_view) - val_size
    
        # --- NLM Denoising Parameters ---
        # Adjust 'h' based on the noise level in mammograms.
        # For medical images, it's crucial to balance noise reduction with detail preservation.
        nlm_h_param = 0.05 # Small h to test, a higher 'h' means more aggressive denoising but also more blurring.
        nlm_patch_size = 7 # Common patch size
        nlm_patch_distance = 15 # Common search window size

        # Define NLM transform instance
        nlm_transform_instance = NLMTransform(
            h=nlm_h_param,
            patch_size=nlm_patch_size,
            patch_distance=nlm_patch_distance,
            multichannel=False # Grayscale mammograms
        )

        # Define transformation pipeline for all data (includes base preprocessing + NLM denoising)
        # This single transformation will be used for training, validation, and test sets.
        base_denoise_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # Add NLM denoising here, after cropping but before converting to Tensor
            nlm_transform_instance,
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0,1])
            # Convert 1 channel to 3 channels here as part of the pipeline
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

        train_and_val_dataset = GrayscaleToResnet50Dataset(self.dataset_train_view, self.dataset_train_pathology_binary_labels)

        # Perform the random split on the indices of the original data.
        # This gives us the indices for the training and validation portions.
        train_indices, val_indices = random_split(
            range(len(train_and_val_dataset)), # Split indices, not the dataset itself
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42) # For reproducibility
        )

        # Extract actual indices from Subset objects
        train_indices = train_indices.indices
        val_indices = val_indices.indices

        # Create the training dataset subset with only denoising
        self.torch_dataset_train_view = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in train_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in train_indices],
            transform=base_denoise_transforms # Apply standard transforms including NLM
        )

        # Create the validation dataset with only denoising
        self.torch_dataset_validation_view = GrayscaleToResnet50Dataset(
            [self.dataset_train_view[i] for i in val_indices],
            [self.dataset_train_pathology_binary_labels[i] for i in val_indices],
            transform=base_denoise_transforms # Apply standard transforms including NLM
        )

        # Create the test dataset with only denoising
        self.torch_dataset_test_view = GrayscaleToResnet50Dataset(
            self.dataset_test_view,
            self.dataset_test_pathology_binary_labels,
            transform=base_denoise_transforms # Apply standard transforms including NLM
        )

        print(f"Training dataset size (denoised only): {len(self.torch_dataset_train_view)}")
        print(f"Validation dataset size (denoised only): {len(self.torch_dataset_validation_view)}")
        print(f"Test dataset size (denoised only): {len(self.torch_dataset_test_view)}")


    def CreateResnet50DatasetCV(self, seed=42):
        """
        Create the pytorch datasets, one for each view, directly from the
        provided train_dataset and test_dataset in __init__.
        No internal train/val split is performed here. Used for cross validation.
        """
        base_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert grayscale to 3 channels for ResNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.406])
        ])

        self.torch_dataset_train_view = GrayscaleToResnet50Dataset(
            self.dataset_train_view, self.dataset_train_pathology_binary_labels, transform=base_transforms
        )
        # Assign the validation dataset here directly
        self.torch_dataset_validation_view = GrayscaleToResnet50Dataset(
            self.dataset_test_view, # This is the K-fold's validation data
            self.dataset_test_pathology_binary_labels, transform=base_transforms
        )
        # This is now explicitly the validation view, so no need for a separate 'test_view'
        self.torch_dataset_test_view = None # Or keep it as a separate concept if needed elsewhere, but not for K-fold val

        print(f"[{self.view_type}] Prepared training dataset size: {len(self.torch_dataset_train_view)}")
        print(f"[{self.view_type}] Prepared validation dataset size: {len(self.torch_dataset_validation_view)}")

    def CreateResnet50DatasetAugmentCV(self):
        """
        Create the PyTorch datasets for a single view with horizontal mirroring augmentation,
        specifically for K-Fold Cross-Validation.
        No internal train/val split is performed here.
        """
        # Define transformation pipeline for training data (includes base + augmentations)
        base_train_eval_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0,1])
            # Convert 1 channel to 3 channels here as part of the pipeline
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

        # Define transformation for horizontal mirroring
        horizontal_flip_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0), # p=1.0 ensures every image is flipped
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
     
        # Create the original training dataset subset (no augmentations, just base preprocessing)
        self.torch_dataset_train_view_original = GrayscaleToResnet50Dataset(
            self.dataset_train_view, # Use the full training data provided to __init__
            self.dataset_train_pathology_binary_labels,
            transform=base_train_eval_transforms # Apply standard transforms
        )

        # Create augmented training dataset subset for horizontal mirroring
        self.torch_dataset_train_view_flipped = GrayscaleToResnet50Dataset(
            self.dataset_train_view, # Use the same raw data for augmentation
            self.dataset_train_pathology_binary_labels,
            transform=horizontal_flip_transform # Apply horizontal flip transform
        )

        # Concatenate all training datasets
        # This creates a training set that is 2x the size of the original training split (original + flipped).
        self.torch_dataset_train_view = ConcatDataset([
            self.torch_dataset_train_view_original,
            self.torch_dataset_train_view_flipped
        ])

        # Create the validation dataset (no augmentations)
        # This uses the `test_dataset` provided to __init__, which is the K-fold's validation split
        self.torch_dataset_validation_view = GrayscaleToResnet50Dataset(
            self.dataset_test_view, # This is the K-fold's validation data
            self.dataset_test_pathology_binary_labels,
            transform=base_train_eval_transforms # Apply standard transforms
        )
        
        self.torch_dataset_test_view = None # Explicitly set to None for clarity in K-fold context
        
        print(f"[{self.view_type} CV Augmented] Original training split size: {len(self.torch_dataset_train_view_original)}")
        print(f"[{self.view_type} CV Augmented] Horizontally flipped augmented training split size: {len(self.torch_dataset_train_view_flipped)}")
        print(f"[{self.view_type} CV Augmented] Total training dataset size (original + augmented): {len(self.torch_dataset_train_view)}")
        print(f"[{self.view_type} CV Augmented] Validation dataset size: {len(self.torch_dataset_validation_view)}")


    def CreateResnet50DatasetAugmentDenoiseCV(self):
        """
        Create the PyTorch datasets for a single view with horizontal mirroring augmentation
        and NLM denoising, specifically for K-Fold Cross-Validation.
        No internal train/val split is performed here.
        """
        # --- NLM Denoising Parameters ---
        nlm_h_param = 0.05 # Example value, adjust as needed (often proportional to noise std dev)
        nlm_patch_size = 7 # Common patch size
        nlm_patch_distance = 15 # Common search window size

        # Define NLM transform instance
        nlm_transform_instance = NLMTransform(
            h=nlm_h_param,
            patch_size=nlm_patch_size,
            patch_distance=nlm_patch_distance,
            multichannel=False # Grayscale mammograms
        )

        # Define transformation pipeline for training data (includes base + augmentations)
        base_train_eval_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            nlm_transform_instance, # Add NLM denoising here
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0,1])
            # Convert 1 channel to 3 channels here as part of the pipeline
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

        # Define transformation for horizontal mirroring
        horizontal_flip_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0), # p=1.0 ensures every image is flipped
            nlm_transform_instance, # Add NLM denoising here for flipped images
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
     
        # Create the original training dataset subset (no augmentations, just base preprocessing)
        self.torch_dataset_train_view_original = GrayscaleToResnet50Dataset(
            self.dataset_train_view, # Use the full training data provided to __init__
            self.dataset_train_pathology_binary_labels,
            transform=base_train_eval_transforms # Apply standard transforms including NLM
        )

        # Create augmented training dataset subset for horizontal mirroring
        self.torch_dataset_train_view_flipped = GrayscaleToResnet50Dataset(
            self.dataset_train_view, # Use the same raw data for augmentation
            self.dataset_train_pathology_binary_labels,
            transform=horizontal_flip_transform # Apply horizontal flip transform including NLM
        )

        # Concatenate all training datasets
        # This creates a training set that is 2x the size of the original training split (original + flipped).
        self.torch_dataset_train_view = ConcatDataset([
            self.torch_dataset_train_view_original,
            self.torch_dataset_train_view_flipped
        ])

        # Create the validation dataset (denoised, unaugmented)
        # This uses the `test_dataset` provided to __init__, which is the K-fold's validation split
        self.torch_dataset_validation_view = GrayscaleToResnet50Dataset(
            self.dataset_test_view, # This is the K-fold's validation data
            self.dataset_test_pathology_binary_labels,
            transform=base_train_eval_transforms # Apply standard transforms including NLM
        )
        
        self.torch_dataset_test_view = None # Explicitly set to None for clarity in K-fold context

        print(f"[{self.view_type} CV Augmented Denoised] Original training split size: {len(self.torch_dataset_train_view_original)}")
        print(f"[{self.view_type} CV Augmented Denoised] Horizontally flipped augmented training split size: {len(self.torch_dataset_train_view_flipped)}")
        print(f"[{self.view_type} CV Augmented Denoised] Total training dataset size (original + augmented): {len(self.torch_dataset_train_view)}")
        print(f"[{self.view_type} CV Augmented Denoised] Validation dataset size: {len(self.torch_dataset_validation_view)}")


    def CreateResnet50DatasetDenoiseCV(self):
        """
        Create the PyTorch datasets for a single view with NLM denoising,
        specifically for K-Fold Cross-Validation.
        No internal train/val split is performed here, and no horizontal mirroring.
        """
        # --- NLM Denoising Parameters ---
        nlm_h_param = 0.05
        nlm_patch_size = 7
        nlm_patch_distance = 15

        # Define NLM transform instance
        nlm_transform_instance = NLMTransform(
            h=nlm_h_param,
            patch_size=nlm_patch_size,
            patch_distance=nlm_patch_distance,
            multichannel=False # Grayscale mammograms
        )

        # Define transformation pipeline for both training and validation data
        # (includes base transforms and NLM denoising, but NO augmentation)
        base_denoise_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            nlm_transform_instance, # Add NLM denoising here
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0,1])
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert 1 channel to 3 channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])
     
        # Create the training dataset (denoised, unaugmented)
        self.torch_dataset_train_view = GrayscaleToResnet50Dataset(
            self.dataset_train_view, # Use the full training data provided to __init__
            self.dataset_train_pathology_binary_labels,
            transform=base_denoise_transforms # Apply standard transforms including NLM
        )

        # Create the validation dataset (denoised, unaugmented)
        # This uses the `test_dataset` provided to __init__, which is the K-fold's validation split
        self.torch_dataset_validation_view = GrayscaleToResnet50Dataset(
            self.dataset_test_view, # This is the K-fold's validation data
            self.dataset_test_pathology_binary_labels,
            transform=base_denoise_transforms # Apply standard transforms including NLM
        )
        
        self.torch_dataset_test_view = None # Explicitly set to None for clarity in K-fold context

        print(f"[{self.view_type} CV Denoised] Prepared training dataset size: {len(self.torch_dataset_train_view)}")
        print(f"[{self.view_type} CV Denoised] Prepared validation dataset size: {len(self.torch_dataset_validation_view)}")
