import numpy as np


def _reproject_error(kpts0_m: np.ndarray, kpts1_m: np.ndarray, H: np.ndarray) -> np.ndarray:
    """매칭 쌍의 symmetric reprojection error 계산."""
    n = len(kpts0_m)
    if n == 0:
        return np.empty(0)

    p0h = np.hstack([kpts0_m, np.ones((n, 1))])
    p0w = (H @ p0h.T).T
    p0w = p0w[:, :2] / p0w[:, 2:3]
    err_fwd = np.linalg.norm(p0w - kpts1_m, axis=1)

    Hinv = np.linalg.inv(H)
    p1h = np.hstack([kpts1_m, np.ones((n, 1))])
    p1w = (Hinv @ p1h.T).T
    p1w = p1w[:, :2] / p1w[:, 2:3]
    err_bwd = np.linalg.norm(p1w - kpts0_m, axis=1)

    return (err_fwd + err_bwd) / 2.0


def mma(kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray,
        H: np.ndarray, thresholds=(1, 3, 5)) -> dict:
    """Mean Matching Accuracy @ 각 threshold (픽셀)."""
    if len(matches) == 0:
        return {f'mma@{t}': 0.0 for t in thresholds}
    m0 = kpts0[matches[:, 0]]
    m1 = kpts1[matches[:, 1]]
    errs = _reproject_error(m0, m1, H)
    return {f'mma@{t}': float((errs < t).mean()) for t in thresholds}
