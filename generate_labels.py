"""
HPatches 전체 이미지 쌍에 대해 GT 라벨 생성.
각 쌍마다 4개 알고리즘의 MMA@3px를 계산하고 normalize 후 argmax로 라벨 결정.
결과: labels/labels.json
"""
import os
import json
import numpy as np
import torch

from eval.hpatches import load_hpatches
from eval.metrics import mma
from matchers.classical import SIFTMatcher, ORBMatcher
from matchers.learned import SPLightGlueMatcher, LoFTRMatcher

MATCHER_NAMES = ['sift', 'orb', 'loftr', 'splg']


def score_pair(pair: dict, matchers: list) -> np.ndarray:
    """이미지 쌍에 대해 각 알고리즘의 MMA@3px 스코어 반환. [4]"""
    scores = []
    for m in matchers:
        try:
            result = m.match(pair['img0'], pair['img1'])
            s = mma(result['kpts0'], result['kpts1'], result['matches'],
                    pair['H'], thresholds=(3,))['mma@3']
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = 'data/hpatches'

    pairs = load_hpatches(data_dir)
    print(f"Loaded {len(pairs)} pairs")

    matchers = [
        SIFTMatcher(),
        ORBMatcher(),
        LoFTRMatcher(device=device),
        SPLightGlueMatcher(device=device),
    ]

    os.makedirs('labels', exist_ok=True)
    records = []

    for i, pair in enumerate(pairs):
        scores = score_pair(pair, matchers)
        norm_scores = normalize(scores)
        label = int(norm_scores.argmax())

        records.append({
            'pair_id':    pair['pair_id'],
            'scene':      pair['scene'],
            'type':       pair['type'],
            'scores':     norm_scores.tolist(),   # [sift, orb, loftr, splg]
            'label':      label,                  # 0~3
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}]")

    with open('labels/labels.json', 'w') as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} labels → labels/labels.json")


if __name__ == '__main__':
    main()
