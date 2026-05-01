import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from src.preprocessing.preprocess import preprocess


class ExperimentDataset(Dataset):
    """
    Dataset for all 8 experiments.

    Args:
        roi_paths: list of paths to precomputed ROI crop .npy files
        labels: list of int labels (0=benign, 1=malignant)
        tda_paths: list of paths to TDA persistence image .npy files (or None)
        use_preprocessing: whether to apply CLAHE + denoise
        augment: whether to apply data augmentation (train only)
        img_size: target image size for ViT
    """

    def __init__(self, roi_paths, labels, tda_paths=None,
                 use_preprocessing=False, augment=False, img_size=224):
        self.roi_paths = roi_paths
        self.labels = labels
        self.tda_paths = tda_paths
        self.use_preprocessing = use_preprocessing
        self.augment = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.roi_paths)

    def __getitem__(self, idx):
        img = np.load(self.roi_paths[idx])

        if self.use_preprocessing:
            img = preprocess(img)

        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.random() > 0.5:
                img = np.flipud(img).copy()

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
        img = torch.from_numpy(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.tda_paths is not None:
            tda = np.load(self.tda_paths[idx]).astype(np.float32).flatten()
            tda = torch.from_numpy(tda)
            return img, tda, label

        return img, label
