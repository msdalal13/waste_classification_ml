import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_features(img):

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog_features = hog(
        gray,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # LBP texture features
    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )

    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # HSV Color histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_features = []

    for channel in range(3):
        hist = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [16],
            [0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()
        color_features.extend(hist)

    # Combine features
    features = np.concatenate([
        hog_features,
        lbp_hist,
        np.array(color_features)
    ])

    return features