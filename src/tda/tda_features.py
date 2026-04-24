import numpy as np
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceImage

def compute_persistence_image(image):
    cp = CubicalPersistence()
    diagrams = cp.fit_transform(image[None, :, :])

    pi = PersistenceImage()
    pi_img = pi.fit_transform(diagrams)

    return pi_img[0]