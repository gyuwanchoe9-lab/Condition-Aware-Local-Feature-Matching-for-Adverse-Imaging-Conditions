import numpy as np
import cv2


# ── Level 1: Detector ────────────────────────────────────────────────────────

def repeatability(kpts0: np.ndarray, kpts1: np.ndarray, H: np.ndarray, eps: float = 3.0) -> float:
    """
    kpts0: (N, 2) [x, y] in image0
    kpts1: (M, 2) [x, y] in image1
    H: 3x3 homography from image0 to image1
    """
    if len(kpts0) == 0 or len(kpts1) == 0:
        return 0.0

    # kpts0를 H로 warp
    kpts0_h = np.hstack([kpts0, np.ones((len(kpts0), 1))])  # (N, 3)
    kpts0_w = (H @ kpts0_h.T).T                             # (N, 3)
    kpts0_w = kpts0_w[:, :2] / kpts0_w[:, 2:3]             # (N, 2)

    # warped kpts0 각각에 대해 kpts1에서 eps 안에 있는 게 있는지
    dists = np.linalg.norm(kpts0_w[:, None] - kpts1[None], axis=2)  # (N, M)
    matched = (dists.min(axis=1) < eps)
    return matched.mean()


def coverage(kpts: np.ndarray, img_h: int, img_w: int, grid: int = 8) -> float:
    """
    Grid cell 균일성. 분산이 낮을수록 고르게 분포.
    반환값: 1 - normalized_std (높을수록 균일)
    """
    if len(kpts) == 0:
        return 0.0
    rows = int(np.ceil(img_h / grid))
    cols = int(np.ceil(img_w / grid))
    counts = np.zeros((rows, cols), dtype=float)
    for x, y in kpts:
        r = min(int(y // grid), rows - 1)
        c = min(int(x // grid), cols - 1)
        counts[r, c] += 1
    std = counts.std()
    mean = counts.mean() + 1e-6
    return float(1.0 - min(std / mean, 1.0))


# ── Level 2 & 3: Matching ────────────────────────────────────────────────────

def _reproject_error(kpts0_m: np.ndarray, kpts1_m: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Symmetric reprojection error for matched pairs."""
    n = len(kpts0_m)
    if n == 0:
        return np.empty(0)

    # Forward: kpts0 → image1
    p0h = np.hstack([kpts0_m, np.ones((n, 1))])
    p0w = (H @ p0h.T).T
    p0w = p0w[:, :2] / p0w[:, 2:3]
    err_fwd = np.linalg.norm(p0w - kpts1_m, axis=1)

    # Backward: kpts1 → image0
    Hinv = np.linalg.inv(H)
    p1h = np.hstack([kpts1_m, np.ones((n, 1))])
    p1w = (Hinv @ p1h.T).T
    p1w = p1w[:, :2] / p1w[:, 2:3]
    err_bwd = np.linalg.norm(p1w - kpts0_m, axis=1)

    return (err_fwd + err_bwd) / 2.0


def mma(kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray,
        H: np.ndarray, thresholds=(1, 3, 5)) -> dict:
    """Mean Matching Accuracy @ each threshold."""
    if len(matches) == 0:
        return {f'mma@{t}': 0.0 for t in thresholds}
    m0 = kpts0[matches[:, 0]]
    m1 = kpts1[matches[:, 1]]
    errs = _reproject_error(m0, m1, H)
    return {f'mma@{t}': float((errs < t).mean()) for t in thresholds}


def maa(kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray,
        H: np.ndarray, thresholds=(1, 3, 5, 10)) -> dict:
    """
    Mean Average Accuracy (AUC of cumulative error distribution).
    LightGlue 논문 방식: corner reprojection error 기반.
    """
    if len(matches) == 0:
        return {f'maa@{t}': 0.0 for t in thresholds}
    m0 = kpts0[matches[:, 0]]
    m1 = kpts1[matches[:, 1]]
    errs = _reproject_error(m0, m1, H)
    errs_sorted = np.sort(errs)
    n = len(errs_sorted)
    result = {}
    for t in thresholds:
        # AUC of cumulative distribution up to threshold t
        xs = np.linspace(0, t, 100)
        ys = np.array([(errs_sorted < x).mean() for x in xs])
        result[f'maa@{t}'] = float(np.trapz(ys, xs) / t)
    return result


def inlier_ratio(kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray,
                 H: np.ndarray, threshold: float = 3.0) -> float:
    if len(matches) == 0:
        return 0.0
    m0 = kpts0[matches[:, 0]]
    m1 = kpts1[matches[:, 1]]
    errs = _reproject_error(m0, m1, H)
    return float((errs < threshold).mean())
