import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from src.utils.image_utils import resolve_mammogram_path, resolve_roi_mask_path, load_dicom, extract_roi_crop

from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceImage

def compute_persistence_image(image):
    cp = CubicalPersistence()
    diagrams = cp.fit_transform(image[None, :, :])

    pi = PersistenceImage()
    pi_img = pi.fit_transform(diagrams)

    return pi_img[0] # Shape is (1, 2, 100, 100)

def precompute_tda_cropped_image(cache_csv_path, tda_dir):
    """Compute persistence images from the pre-cropped lesions."""
    os.makedirs(tda_dir, exist_ok=True)
    cache = pd.read_csv(cache_csv_path)
    if 'tda_crop_path' in cache.columns and cache['tda_crop_path'].notna().all():
        print(f'TDA crop already computed for {tda_dir}, skipping.')
        return cache
    
    tda_paths = []
    for i, row in tqdm(cache.iterrows(), total=len(cache)):
        tda_path = os.path.join(tda_dir, f'tda_{i}_pi.npy')
        if not os.path.exists(tda_path):
            try:
                img = np.load(row['path'])
                pi = compute_persistence_image(img)
                np.save(tda_path, pi)
            except Exception as e:
                print(f'SKIP {row}: {e}')
                np.save(tda_path, np.zeros((1,)))
        tda_paths.append(tda_path)
    cache['tda_crop_path'] = tda_paths
    cache.to_csv(cache_csv_path, index=False)
    return cache

def precompute_tda_masked_mammography(cache_csv_path, df, tda_dir):
    """Compute persistence images from masked crops (full mammogram x ROI mask)."""
    os.makedirs(tda_dir, exist_ok=True)
    cache = pd.read_csv(cache_csv_path)
    if 'tda_masked_path' in cache.columns and cache['tda_masked_path'].notna().all():
        print(f'TDA masked already computed for {tda_dir}, skipping.')
        return cache
    
    tda_paths = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        tda_path = os.path.join(tda_dir, f'tda_{i}_pi_masked.npy')
        if not os.path.exists(tda_path):
            try:
                mammogram = load_dicom(resolve_mammogram_path(row['image file path']))
                mask = load_dicom(resolve_roi_mask_path(row['ROI mask file path']))
                masked_crop = extract_roi_crop(mammogram, mask)
                pi = compute_persistence_image(masked_crop)
                np.save(tda_path, pi)
            except Exception as e:
                print(f'SKIP {row}: {e}')
                np.save(tda_path, np.zeros((1,)))
        tda_paths.append(tda_path)
    cache['tda_masked_path'] = tda_paths[:len(cache)]
    cache.to_csv(cache_csv_path, index=False)
    return cache