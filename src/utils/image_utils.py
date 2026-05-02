import pydicom
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

BASE_PATH = r"C:/Users/esthe/Documents/UOC/TFM/TDA_Visual_Transformer/data/raw/cbis_ddsm"

def load_dicom(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)

    # normalize to 0–255
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    return img

def get_all_dcm_files_with_size(rel_path):
    rel_path = rel_path.strip()
    folder = os.path.dirname(rel_path)

    full_folder = os.path.join(BASE_PATH, folder)

    full_folder = r"\\?\{}".format(os.path.abspath(full_folder))

    if not os.path.exists(full_folder):
        raise FileNotFoundError(f"Folder not found: {full_folder}")

    dcm_files = [f for f in os.listdir(full_folder) if f.endswith(".dcm")]

    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files in {full_folder}")

    files_with_size = []
    for f in dcm_files:
        p = os.path.join(full_folder, f)
        files_with_size.append((p, os.path.getsize(p)))

    files_with_size.sort(key=lambda x: x[1], reverse=False)

    return files_with_size

def resolve_mammogram_path(rel_path):
    files_with_size = get_all_dcm_files_with_size(rel_path)

    if(len(files_with_size) != 1):
        raise Exception("Unexpected amount of files")
    cropped_mammogram_path = files_with_size[0][0] 

    return cropped_mammogram_path

def resolve_cropped_mammogram_path(rel_path):
    files_with_size = get_all_dcm_files_with_size(rel_path)

    if(len(files_with_size) != 1 and len(files_with_size) != 2):
        raise Exception("Unexpected amount of files")
    
    cropped_mammogram_path = files_with_size[0][0] # smallest = mammogram

    return cropped_mammogram_path

def resolve_roi_mask_path(rel_path):
    files_with_size = get_all_dcm_files_with_size(rel_path)

    if(len(files_with_size) != 1 and len(files_with_size) != 2):
        raise Exception("Unexpected amount of files")
    
    mask_path = None
    if len(files_with_size) == 2:
        mask_path = files_with_size[1][0] # biggest = ROI mask
    else:
        mask_path = files_with_size[0][0]

    return mask_path

def cache_cropped_mammogram_images_as_np_arrays(save_dir, df):
    os.makedirs(save_dir, exist_ok=True)

    image_paths = []
    labels = []
    cached_paths = []

    for i, row in tqdm(df.iterrows()):
        try:
            full_path = resolve_cropped_mammogram_path(row["cropped image file path"])

            label = 1 if row["pathology"] == "MALIGNANT" else 0

            image_paths.append(full_path)
            labels.append(label)

            try:
                img = load_dicom(full_path)

                save_path = os.path.join(save_dir, f"img_{i}.npy")
                np.save(save_path, img)

                cached_paths.append(save_path)

            except Exception as e:
                print("Error:", e)

        except Exception as e:
            print("Skipping:", e)
            
        row

    cache_df = pd.DataFrame({
        "path": cached_paths,
        "label": labels
    })

    cache_df.to_csv(os.path.join(save_dir, f"cached.csv"), index=False)

def extract_roi_crop(mammogram, mask, padding_ratio=0.15):
    """
    Extract the ROI crop from a full mammogram using the binary mask.
    
    Returns:
        masked_crop: cropped mammogram region with mask applied
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    pad_h = int((rmax - rmin) * padding_ratio)
    pad_w = int((cmax - cmin) * padding_ratio)
    
    rmin = max(0, rmin - pad_h)
    rmax = min(mammogram.shape[0], rmax + pad_h)
    cmin = max(0, cmin - pad_w)
    cmax = min(mammogram.shape[1], cmax + pad_w)
    
    roi_crop = mammogram[rmin:rmax, cmin:cmax]
    mask_crop = mask[rmin:rmax, cmin:cmax]
    masked_crop = roi_crop * (mask_crop > 0).astype(np.float32)
    
    return masked_crop