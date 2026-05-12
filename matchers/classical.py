import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _kpts_to_array(kpts) -> np.ndarray:
    if len(kpts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[k.pt[0], k.pt[1]] for k in kpts], dtype=np.float32)


def _assign_descs_by_nearest(query_kpts: np.ndarray, ref_kpts: np.ndarray, ref_descs: np.ndarray) -> np.ndarray:
    """query_kpts 각각에 대해 ref_kpts 중 가장 가까운 keypoint의 descriptor를 할당."""
    if len(query_kpts) == 0 or len(ref_kpts) == 0:
        dim = ref_descs.shape[1] if ref_descs.ndim == 2 else 256
        return np.zeros((len(query_kpts), dim), dtype=np.float32)
    q = query_kpts.reshape(-1, 2)
    r = ref_kpts.reshape(-1, 2)
    dists = np.linalg.norm(q[:, None] - r[None], axis=2)  # (N, M)
    nn_idx = dists.argmin(axis=1)
    return ref_descs[nn_idx]


def _mutual_nn_match(desc0, desc1, norm=cv2.NORM_L2):
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(desc0, desc1)
    if len(matches) == 0:
        return np.empty((0, 2), dtype=int)
    matches = sorted(matches, key=lambda m: m.distance)
    return np.array([[m.queryIdx, m.trainIdx] for m in matches], dtype=int)


class SIFTMatcher:
    def __init__(self, n_features=2000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def detect(self, img: np.ndarray) -> np.ndarray:
        kpts = self.sift.detect(_to_gray(img))
        return _kpts_to_array(kpts)

    def describe(self, img: np.ndarray, kpts_cv):
        _, descs = self.sift.compute(_to_gray(img), kpts_cv)
        return descs

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        gray0, gray1 = _to_gray(img0), _to_gray(img1)
        kpts0, descs0 = self.sift.detectAndCompute(gray0, None)
        kpts1, descs1 = self.sift.detectAndCompute(gray1, None)
        if descs0 is None or descs1 is None:
            return {'kpts0': np.empty((0,2)), 'kpts1': np.empty((0,2)), 'matches': np.empty((0,2), dtype=int)}
        matches = _mutual_nn_match(descs0, descs1)
        return {
            'kpts0': _kpts_to_array(kpts0),
            'kpts1': _kpts_to_array(kpts1),
            'matches': matches,
        }


class ORBMatcher:
    def __init__(self, n_features=2000):
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def detect(self, img: np.ndarray) -> np.ndarray:
        kpts = self.orb.detect(_to_gray(img))
        return _kpts_to_array(kpts)

    def describe(self, img: np.ndarray, kpts_cv):
        _, descs = self.orb.compute(_to_gray(img), kpts_cv)
        return descs

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        gray0, gray1 = _to_gray(img0), _to_gray(img1)
        kpts0, descs0 = self.orb.detectAndCompute(gray0, None)
        kpts1, descs1 = self.orb.detectAndCompute(gray1, None)
        if descs0 is None or descs1 is None:
            return {'kpts0': np.empty((0,2)), 'kpts1': np.empty((0,2)), 'matches': np.empty((0,2), dtype=int)}
        matches = _mutual_nn_match(descs0, descs1, norm=cv2.NORM_HAMMING)
        return {
            'kpts0': _kpts_to_array(kpts0),
            'kpts1': _kpts_to_array(kpts1),
            'matches': matches,
        }


class CrossCombinationMatcher:
    """
    detector: 'sp' | 'sift'
    descriptor: 'sp' | 'sift'
    SP descriptor는 learned.py의 SuperPointMatcher에서 주입받음.
    """
    def __init__(self, detector: str, descriptor: str, sp_matcher=None, n_features=2000):
        assert detector in ('sp', 'sift') and descriptor in ('sp', 'sift')
        self.detector = detector
        self.descriptor = descriptor
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.sp = sp_matcher  # SuperPointMatcher instance (descriptor='sp'일 때 필요)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        # 1) detect
        if self.detector == 'sift':
            kpts0_cv = self.sift.detect(_to_gray(img0))
            kpts1_cv = self.sift.detect(_to_gray(img1))
            kpts0_arr = _kpts_to_array(kpts0_cv)
            kpts1_arr = _kpts_to_array(kpts1_cv)
        else:  # sp
            kpts0_arr = self.sp.detect(_to_gray(img0))
            kpts1_arr = self.sp.detect(_to_gray(img1))
            kpts0_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts0_arr]
            kpts1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts1_arr]

        # 2) describe
        if self.descriptor == 'sift':
            _, descs0 = self.sift.compute(_to_gray(img0), kpts0_cv)
            _, descs1 = self.sift.compute(_to_gray(img1), kpts1_cv)
            norm = cv2.NORM_L2
        else:  # sp: SP keypoint 위치에서 descriptor 추출 후 nearest-neighbor로 입력 kpts에 할당
            out0 = self.sp._extract(_to_gray(img0))
            out1 = self.sp._extract(_to_gray(img1))
            sp_kpts0 = out0['keypoints'][0].cpu().numpy()
            sp_descs0 = out0['descriptors'][0].cpu().numpy()
            sp_kpts1 = out1['keypoints'][0].cpu().numpy()
            sp_descs1 = out1['descriptors'][0].cpu().numpy()
            descs0 = _assign_descs_by_nearest(kpts0_arr, sp_kpts0, sp_descs0)
            descs1 = _assign_descs_by_nearest(kpts1_arr, sp_kpts1, sp_descs1)
            norm = cv2.NORM_L2

        if descs0 is None or descs1 is None or len(descs0) == 0 or len(descs1) == 0:
            return {'kpts0': kpts0_arr, 'kpts1': kpts1_arr, 'matches': np.empty((0,2), dtype=int)}

        matches = _mutual_nn_match(descs0.astype(np.float32), descs1.astype(np.float32), norm)
        return {'kpts0': kpts0_arr, 'kpts1': kpts1_arr, 'matches': matches}
