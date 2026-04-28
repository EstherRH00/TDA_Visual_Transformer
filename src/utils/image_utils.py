import pydicom
import numpy as np
import os

BASE_PATH = r"C:/Users/esthe/Documents/UOC/TFM/TDA_Visual_Transformer/data/raw/cbis_ddsm"

def load_dicom(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)

    # normalize to 0–255
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    return img

def resolve_dicom_path(rel_path):
    rel_path = rel_path.strip()
    folder = os.path.dirname(rel_path)

    full_folder = os.path.join(BASE_PATH, folder)

    full_folder = r"\\?\{}".format(os.path.abspath(full_folder))

    if not os.path.exists(full_folder):
        raise FileNotFoundError(f"Folder not found: {full_folder}")

    dcm_files = [f for f in os.listdir(full_folder) if f.endswith(".dcm")]

    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files in {full_folder}")

    # pick cropped image
    files_with_size = []
    for f in dcm_files:
        p = os.path.join(full_folder, f)
        files_with_size.append((p, os.path.getsize(p)))

    files_with_size.sort(key=lambda x: x[1], reverse=False)

    if len(files_with_size) > 1:
        print("⚠️ Multiple DICOM files:")
        for p, s in files_with_size:
            print(f"  {os.path.basename(p)} → {s/1e6:.2f} MB")

    selected_path = files_with_size[0][0]

    print("SELECTED:", selected_path)
    print("EXISTS:", os.path.exists(selected_path))

    return selected_path