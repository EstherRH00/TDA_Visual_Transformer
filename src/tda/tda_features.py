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
    return pi_img[0]

import gudhi as gd
import gudhi.representations

def compute_vector_descriptors(image):
    """Compute compact TDA vector descriptors: Betti curves + Landscapes + Silhouettes for H0 and H1.
    Returns a 1D numpy array of ~1200 dimensions."""
    cc = gd.CubicalComplex(dimensions=image.shape, top_dimensional_cells=image.flatten().astype(float))
    cc.persistence()

    persistence_0 = [p for p in cc.persistence_intervals_in_dimension(0) if p[1] != float('inf')]
    persistence_1 = [p for p in cc.persistence_intervals_in_dimension(1) if p[1] != float('inf')]

    resolution = 100
    n_landscapes = 5

    bc_0 = bc_1 = np.zeros(resolution)
    lc_0 = lc_1 = np.zeros(resolution * n_landscapes)
    s_0 = s_1 = np.zeros(resolution)

    if len(persistence_0) > 0:
        p0 = np.array(persistence_0)
        bc_0 = gd.representations.BettiCurve(resolution=resolution)(p0).flatten()
        lc_0 = gd.representations.Landscape(num_landscapes=n_landscapes, resolution=resolution)(p0).flatten()
        s_0 = gd.representations.Silhouette(resolution=resolution)(p0).flatten()

    if len(persistence_1) > 0:
        p1 = np.array(persistence_1)
        bc_1 = gd.representations.BettiCurve(resolution=resolution)(p1).flatten()
        lc_1 = gd.representations.Landscape(num_landscapes=n_landscapes, resolution=resolution)(p1).flatten()
        s_1 = gd.representations.Silhouette(resolution=resolution)(p1).flatten()

    return np.concatenate([bc_0, bc_1, lc_0, lc_1, s_0, s_1])

def precompute_tda_vector_descriptors_cropped(cache_csv_path, tda_dir):
    """Compute vector descriptors from pre-cropped lesions."""
    os.makedirs(tda_dir, exist_ok=True)
    cache = pd.read_csv(cache_csv_path)
    if 'tda_vec_crop_path' in cache.columns and cache['tda_vec_crop_path'].notna().all():
        print(f'TDA vector crop already computed for {tda_dir}, skipping.')
        return cache
    tda_paths = []
    for i, row in tqdm(cache.iterrows(), total=len(cache)):
        tda_path = os.path.join(tda_dir, f'tda_{i}_vec.npy')
        if not os.path.exists(tda_path):
            try:
                img = np.load(row['path'])
                vec = compute_vector_descriptors(img)
                np.save(tda_path, vec)
            except Exception as e:
                print(f'SKIP {i}: {e}')
                np.save(tda_path, np.zeros((1200,)))
        tda_paths.append(tda_path)
    cache['tda_vec_crop_path'] = tda_paths
    cache.to_csv(cache_csv_path, index=False)
    return cache

def precompute_tda_vector_descriptors_masked(cache_csv_path, df, tda_dir):
    """Compute vector descriptors from masked crops."""
    os.makedirs(tda_dir, exist_ok=True)
    cache = pd.read_csv(cache_csv_path)
    if 'tda_vec_masked_path' in cache.columns and cache['tda_vec_masked_path'].notna().all():
        print(f'TDA vector masked already computed for {tda_dir}, skipping.')
        return cache
    tda_paths = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        tda_path = os.path.join(tda_dir, f'tda_{i}_vec_masked.npy')
        if not os.path.exists(tda_path):
            try:
                mammogram = load_dicom(resolve_mammogram_path(row['image file path']))
                mask = load_dicom(resolve_roi_mask_path(row['ROI mask file path']))
                masked_crop = extract_roi_crop(mammogram, mask)
                vec = compute_vector_descriptors(masked_crop.astype(np.uint8))
                np.save(tda_path, vec)
            except Exception as e:
                print(f'SKIP {i}: {e}')
                np.save(tda_path, np.zeros((1200,)))
        tda_paths.append(tda_path)
    cache['tda_vec_masked_path'] = tda_paths[:len(cache)]
    cache.to_csv(cache_csv_path, index=False)
    return cache

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

def precompute_tda_masked_mammogram(cache_csv_path, df, tda_dir):
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