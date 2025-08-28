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


class ClassifierHead(nn.Module):
    """
        Base class for the final classifier part 
    """
    def __init__(self, in_features):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(in_features, 1)
        # Initialize new weights (more information at https://docs.pytorch.org/docs/stable/nn.init.html)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)