import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np  
from src.utils.image_utils import load_dicom

class MammographyDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_dicom(self.image_paths[idx])

        img = cv2.resize(img, (224, 224))

        # Convert to 3 channels (VERY IMPORTANT for ViT)
        img = np.stack([img, img, img], axis=-1)

        img = torch.tensor(img).float().permute(2,0,1) / 255.0

        label = torch.tensor(self.labels[idx]).float()

        return img, label