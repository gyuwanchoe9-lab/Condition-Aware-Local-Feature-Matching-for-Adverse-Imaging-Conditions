import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _kpts_to_array(kpts) -> np.ndarray:
    if len(kpts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[k.pt[0], k.pt[1]] for k in kpts], dtype=np.float32)


def _mutual_nn_match(desc0, desc1, norm=cv2.NORM_L2):
    """Brute-force mutual nearest neighbor matching."""
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(desc0, desc1)
    if len(matches) == 0:
        return np.empty((0, 2), dtype=int)
    matches = sorted(matches, key=lambda m: m.distance)
    return np.array([[m.queryIdx, m.trainIdx] for m in matches], dtype=int)


class SIFTMatcher:
    def __init__(self, n_features=2000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        """SIFT detect+describe+match. 반환: kpts0, kpts1, matches."""
        kpts0, descs0 = self.sift.detectAndCompute(_to_gray(img0), None)
        kpts1, descs1 = self.sift.detectAndCompute(_to_gray(img1), None)
        if descs0 is None or descs1 is None:
            return {'kpts0': np.empty((0, 2)), 'kpts1': np.empty((0, 2)), 'matches': np.empty((0, 2), dtype=int)}
        return {
            'kpts0': _kpts_to_array(kpts0),
            'kpts1': _kpts_to_array(kpts1),
            'matches': _mutual_nn_match(descs0, descs1),
        }


class ORBMatcher:
    def __init__(self, n_features=2000):
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        """ORB detect+describe+match. 반환: kpts0, kpts1, matches."""
        kpts0, descs0 = self.orb.detectAndCompute(_to_gray(img0), None)
        kpts1, descs1 = self.orb.detectAndCompute(_to_gray(img1), None)
        if descs0 is None or descs1 is None:
            return {'kpts0': np.empty((0, 2)), 'kpts1': np.empty((0, 2)), 'matches': np.empty((0, 2), dtype=int)}
        return {
            'kpts0': _kpts_to_array(kpts0),
            'kpts1': _kpts_to_array(kpts1),
            'matches': _mutual_nn_match(descs0, descs1, norm=cv2.NORM_HAMMING),
        }
