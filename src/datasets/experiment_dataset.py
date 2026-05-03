import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from ..utils.preprocess import preprocess


class ExperimentDataset(Dataset):
    """PyTorch Dataset that loads precomputed .npy images with optional preprocessing,
    augmentation, and TDA features for all experiments (E1–E8).

    Args:
        - roi_paths: list of paths to precomputed ROI crop .npy files.
        - labels: list of int labels (0=benign, 1=malignant).
        - tda_paths: list of paths to precomputed TDA .npy files, or None.
        - use_preprocessing: if True, apply CLAHE + Gaussian denoise.
        - augment: if True, apply basic augmentation (random horizontal/vertical flips).
        - aggressive_augmentation: if True, also apply rotation ±15°, zoom 85–100%, shear ±0.1.
        - tda_as_image: if True, keep TDA array as multi-dim (for DualViTFusionModel);
          if False, flatten to 1D vector (for FusionModel).
        - img_size: target image size for ViT input (default 224).
    """

    def __init__(self, image_paths, labels, tda_paths=None,
                 use_preprocessing=False, augment=False, aggressive_augmentation=False,
                 tda_as_image=False, img_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.tda_paths = tda_paths
        self.use_preprocessing = use_preprocessing
        self.augment = augment
        self.aggressive_augmentation = aggressive_augmentation
        self.tda_as_image = tda_as_image
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx])

        if self.use_preprocessing:
            img = preprocess(img)

        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.random() > 0.5:
                img = np.flipud(img).copy()

        if self.aggressive_augmentation:
            # Random rotation (±15 degrees, as in Paper 1)
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # Random zoom/crop (0.85–1.0 of original, then resize back)
            scale = np.random.uniform(0.85, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            img = img[top:top + new_h, left:left + new_w]
            img = cv2.resize(img, (w, h))
            # Random shear (small, ±0.1)
            if np.random.random() > 0.5:
                shear = np.random.uniform(-0.1, 0.1)
                M_shear = np.float32([[1, shear, 0], [0, 1, 0]])
                img = cv2.warpAffine(img, M_shear, (w, h), borderMode=cv2.BORDER_REFLECT)

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
        img = torch.from_numpy(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.tda_paths is not None:
            tda = np.load(self.tda_paths[idx]).astype(np.float32)
            if not self.tda_as_image:
                tda = tda.flatten()
            tda = torch.from_numpy(tda)
            return img, tda, label

        return img, label
