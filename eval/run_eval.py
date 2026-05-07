import time
import numpy as np
import cv2
from typing import Dict, List

from .metrics import repeatability, coverage, mma, maa, inlier_ratio


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ── Level 1: Detector ────────────────────────────────────────────────────────

def eval_detector(matcher, pairs: List[Dict]) -> Dict:
    """
    matcher.detect(img_gray) → kpts (N, 2) 가 구현된 객체 필요.
    """
    rep_list, cov_list = [], []
    for p in pairs:
        g0, g1 = _to_gray(p['img0']), _to_gray(p['img1'])
        kpts0 = matcher.detect(g0)
        kpts1 = matcher.detect(g1)
        rep_list.append(repeatability(kpts0, kpts1, p['H']))
        h, w = g0.shape[:2]
        cov_list.append(coverage(kpts0, h, w))

    return {
        'repeatability': float(np.mean(rep_list)),
        'coverage':      float(np.mean(cov_list)),
    }


# ── Level 2 & 3: Full Pipeline ───────────────────────────────────────────────

def eval_pipeline(matcher, pairs: List[Dict]) -> Dict:
    """
    matcher.match(img0, img1) → dict(kpts0, kpts1, matches) 가 구현된 객체 필요.
    """
    mma1, mma3, mma5 = [], [], []
    maa1, maa3, maa5, maa10 = [], [], [], []
    inliers, runtimes = [], []

    for p in pairs:
        t0 = time.perf_counter()
        out = matcher.match(p['img0'], p['img1'])
        runtimes.append((time.perf_counter() - t0) * 1000)

        kpts0, kpts1, matches = out['kpts0'], out['kpts1'], out['matches']
        H = p['H']

        m = mma(kpts0, kpts1, matches, H)
        mma1.append(m['mma@1']); mma3.append(m['mma@3']); mma5.append(m['mma@5'])

        a = maa(kpts0, kpts1, matches, H)
        maa1.append(a['maa@1']); maa3.append(a['maa@3'])
        maa5.append(a['maa@5']); maa10.append(a['maa@10'])

        inliers.append(inlier_ratio(kpts0, kpts1, matches, H))

    return {
        'mma@1':        float(np.mean(mma1)),
        'mma@3':        float(np.mean(mma3)),
        'mma@5':        float(np.mean(mma5)),
        'maa@1':        float(np.mean(maa1)),
        'maa@3':        float(np.mean(maa3)),
        'maa@5':        float(np.mean(maa5)),
        'maa@10':       float(np.mean(maa10)),
        'inlier_ratio': float(np.mean(inliers)),
        'runtime_ms':   float(np.mean(runtimes)),
    }
