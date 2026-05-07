import cv2
import numpy as np


def compute_features(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    brightness = gray.mean() / 255.0
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {'brightness': brightness, 'laplacian_var': lap_var}


def classify(img: np.ndarray, tau_b: float = 0.3, tau_l: float = 50.0) -> str:
    """Returns 'degraded' or 'normal'. tau values tuned on HPatches val split."""
    feats = compute_features(img)
    if feats['brightness'] < tau_b or feats['laplacian_var'] < tau_l:
        return 'degraded'
    return 'normal'
