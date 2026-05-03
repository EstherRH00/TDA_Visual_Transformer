import cv2

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast.

    Args:
        - image: uint8 grayscale numpy array.

    Returns:
        - image: contrast-enhanced uint8 grayscale numpy array.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def denoise(image):
    """Apply Gaussian blur with a 5×5 kernel to reduce noise.

    Args:
        - image: uint8 grayscale numpy array.

    Returns:
        - image: denoised uint8 grayscale numpy array.
    """
    return cv2.GaussianBlur(image, (5,5), 0)

def preprocess(image):
    """Apply the full preprocessing pipeline: CLAHE followed by Gaussian denoise.

    Args:
        - image: uint8 grayscale numpy array.

    Returns:
        - image: preprocessed uint8 grayscale numpy array.
    """
    image = apply_clahe(image)
    image = denoise(image)
    return image