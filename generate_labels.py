"""
HPatches 전체 이미지 쌍에 대해 GT 라벨 생성.
각 쌍마다 brightness / blur / noise 증강을 따로 적용해 3개 샘플 생성.
각 샘플에서 4개 알고리즘 MMA@3px → normalize → argmax → 라벨.
결과: labels/labels.json (총 580 × 3 = 1,740 샘플)
"""
import os
import json
import random
import cv2
import numpy as np
import torch

from eval.hpatches import load_hpatches
from eval.metrics import mma
from iqa import measure_conditions
from matchers.classical import SIFTMatcher, ORBMatcher
from matchers.learned import SPLightGlueMatcher, LoFTRMatcher

MATCHER_NAMES = ['sift', 'orb', 'loftr', 'splg']


# ── 증강 함수 ─────────────────────────────────────────────────────────────────

def aug_brightness(img: np.ndarray) -> np.ndarray:
    """랜덤 밝기 감소 (factor 0.2~0.7)."""
    factor = random.uniform(0.2, 0.7)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def aug_motion_blur(img: np.ndarray) -> np.ndarray:
    """랜덤 방향/크기의 motion blur 커널 적용."""
    size = random.choice([9, 15, 21])
    angle = random.uniform(0, 180)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def aug_noise(img: np.ndarray) -> np.ndarray:
    """랜덤 sigma의 Gaussian noise 추가."""
    sigma = random.uniform(20, 60)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


AUGMENTS = {
    'brightness': aug_brightness,
    'blur':       aug_motion_blur,
    'noise':      aug_noise,
}


# ── 매칭 스코어 ───────────────────────────────────────────────────────────────

def score_pair(img0, img1, H, matchers) -> np.ndarray:
    """이미지 쌍에 대해 각 알고리즘의 MMA@3px 스코어 반환. [4]"""
    scores = []
    for m in matchers:
        try:
            result = m.match(img0, img1)
            s = mma(result['kpts0'], result['kpts1'], result['matches'],
                    H, thresholds=(3,))['mma@3']
        except Exception:
            s = 0.0
        scores.append(s)
    return np.array(scores, dtype=np.float32)


def normalize(scores: np.ndarray) -> np.ndarray:
    """합이 1이 되도록 정규화. 전부 0이면 uniform 반환."""
    total = scores.sum()
    if total < 1e-6:
        return np.ones(4, dtype=np.float32) / 4
    return scores / total


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pairs = load_hpatches('data/hpatches')
    print(f"Loaded {len(pairs)} pairs → {len(pairs) * 3} samples after augmentation")

    matchers = [
        SIFTMatcher(),
        ORBMatcher(),
        LoFTRMatcher(device=device),
        SPLightGlueMatcher(device=device),
    ]

    os.makedirs('labels', exist_ok=True)
    records = []
    total = len(pairs) * len(AUGMENTS)

    for i, pair in enumerate(pairs):
        for aug_name, aug_fn in AUGMENTS.items():
            # img0에만 증강 적용 (조건 측정 기준)
            img0_aug = aug_fn(pair['img0'])
            img1_aug = aug_fn(pair['img1'])

            scores = score_pair(img0_aug, img1_aug, pair['H'], matchers)
            norm_scores = normalize(scores)
            label = int(norm_scores.argmax())

            # 증강된 img0에서 조건 수치 측정
            conditions = measure_conditions(img0_aug)

            records.append({
                'pair_id':    pair['pair_id'],
                'scene':      pair['scene'],
                'type':       pair['type'],
                'aug':        aug_name,               # 적용된 증강 종류
                'conditions': conditions,             # {brightness, blur, noise}
                'scores':     norm_scores.tolist(),
                'label':      label,
            })

        done = (i + 1) * len(AUGMENTS)
        if done % 60 == 0:
            print(f"  [{done}/{total}]")

    with open('labels/labels.json', 'w') as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} labels → labels/labels.json")


if __name__ == '__main__':
    main()
