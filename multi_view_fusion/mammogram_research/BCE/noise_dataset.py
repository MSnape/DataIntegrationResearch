import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class RandomNoiseDataset(Dataset):
    """
    Dataset that generates random noise images with user-provided or random binary labels.
    The images are compatible with a ResNet50 model.
    """
    def __init__(self, num_images: int, image_size: int = 224, num_channels: int = 3, labels: torch.Tensor = None):
        """
        Initializes the RandomNoiseDataset.

        Args:
            num_images (int): Number of random images to generate.
            image_size (int): The height and width of the square images (e.g., 224 for ResNet50).
            num_channels (int): The number of color channels (e.g., 3 for RGB).
            labels (torch.Tensor, optional): A 1D PyTorch tensor of labels (0 or 1) for the images.
                                            If None, labels will be randomly generated.
        """
        if not isinstance(num_images, int) or num_images <= 0:
            raise ValueError("num_images must be a positive integer.")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if not isinstance(num_channels, int) or num_channels <= 0:
            raise ValueError("num_channels must be a positive integer.")

        self.num_images = num_images
        self.image_size = image_size
        self.num_channels = num_channels

        # Generate random noise images:
        # Use torch.rand to create tensors uniformly sampled from [0, 1).
        # Shape: (num_images, num_channels, image_size, image_size)
        self.image_arrays = torch.rand(self.num_images, self.num_channels, self.image_size, self.image_size, dtype=torch.float32)

        # Assign labels 
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError("Provided labels must be a torch.Tensor.")
            if labels.ndim != 1 or labels.shape[0] != num_images:
                raise ValueError(f"Provided labels tensor must be 1D and have {num_images} elements. Given shape {labels.shape}.")
            self.labels = labels.long() 
        else:
            # Generate random binary labels (0 or 1)
            self.labels = torch.randint(0, 2, (self.num_images,), dtype=torch.long)

        # Tranformation for images, normalization for ResNet-50 ImageNet weights
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"RandomNoiseDataset created with {num_images} images.")
        print(f"Image shape: ({num_channels}, {image_size}, {image_size})")
        print(f"Label distribution (approx): {torch.sum(self.labels == 0).item()} zeros, {torch.sum(self.labels == 1).item()} ones.")


    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.
        """
        return self.num_images

    def __getitem__(self, idx: int):
        """
        Retrieves an image and its corresponding label by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the transformed image tensor and its label.
        """
        if idx >= self.num_images or idx < 0:
            raise IndexError("Index out of bounds for the dataset.")

        image = self.image_arrays[idx]
        label = self.labels[idx]

        # Apply transformation
        image = self.transform(image)

        return image, label