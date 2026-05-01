import cv2

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def denoise(image):
    return cv2.GaussianBlur(image, (5,5), 0)

def preprocess(image):
    image = apply_clahe(image)
    image = denoise(image)
    return image