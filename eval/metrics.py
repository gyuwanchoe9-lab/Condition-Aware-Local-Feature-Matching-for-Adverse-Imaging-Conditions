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


def auc(kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray,
        H: np.ndarray, thresholds=(3, 5, 10)) -> dict:
    """AUC of cumulative error distribution @ 각 threshold (픽셀)."""
    if len(matches) == 0:
        return {f'auc@{t}': 0.0 for t in thresholds}
    m0 = kpts0[matches[:, 0]]
    m1 = kpts1[matches[:, 1]]
    errs = np.sort(_reproject_error(m0, m1, H))
    result = {}
    for t in thresholds:
        xs = np.linspace(0, t, 100)
        ys = np.array([(errs < x).mean() for x in xs])
        result[f'auc@{t}'] = float(np.trapz(ys, xs) / t)
    return result


def repeatability(kpts0: np.ndarray, kpts1: np.ndarray,
                  H: np.ndarray, eps: float = 3.0) -> float:
    """
    img0 keypoint를 H로 변환 후 img1 keypoint와 eps 이내인 비율.
    detector-free 방식(LoFTR)에는 적용 불가.
    """
    if len(kpts0) == 0 or len(kpts1) == 0:
        return 0.0
    kpts0_h = np.hstack([kpts0, np.ones((len(kpts0), 1))])
    kpts0_w = (H @ kpts0_h.T).T
    kpts0_w = kpts0_w[:, :2] / kpts0_w[:, 2:3]
    dists = np.linalg.norm(kpts0_w[:, None] - kpts1[None], axis=2)
    return float((dists.min(axis=1) < eps).mean())
