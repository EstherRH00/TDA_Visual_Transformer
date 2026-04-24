import os
import cv2
import torch
from torch.utils.data import Dataset

class MammographyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, tda_paths=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.tda_paths = tda_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            img = self.transform(img)

        img = torch.tensor(img).float().unsqueeze(0) / 255.0

        if self.tda_paths:
            tda = torch.tensor(np.load(self.tda_paths[idx])).float()
            return img, tda, self.labels[idx]

        return img, self.labels[idx]