"""
4개 매칭 알고리즘을 HPatches 원본 데이터셋에서 평가.
- MMA@1, @3, @5px
- AUC@3, @5, @10px
- Repeatability@3px (SIFT, ORB, SP+LG만 / LoFTR 제외)
결과: results/matcher_eval.txt, results/matcher_eval.json
"""
import json
import os
import numpy as np
import torch
import cv2

from eval.hpatches import load_hpatches
from eval.metrics import mma, auc, repeatability
from matchers.classical import SIFTMatcher, ORBMatcher
from matchers.learned import SPLightGlueMatcher, LoFTRMatcher

MATCHER_NAMES = ['SIFT', 'ORB', 'SP+LG', 'LoFTR']


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _kpts_to_array(kpts) -> np.ndarray:
    if len(kpts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[k.pt[0], k.pt[1]] for k in kpts], dtype=np.float32)


def detect_keypoints(name: str, img: np.ndarray, matchers: dict) -> np.ndarray:
    """알고리즘별 keypoint 검출 (Repeatability 계산용)."""
    if name == 'SIFT':
        kpts = matchers['SIFT'].sift.detect(_to_gray(img))
        return _kpts_to_array(kpts)
    elif name == 'ORB':
        kpts = matchers['ORB'].orb.detect(_to_gray(img))
        return _kpts_to_array(kpts)
    elif name == 'SP+LG':
        from lightglue.utils import rbd
        import torch
        gray = _to_gray(img)
        t = torch.from_numpy(gray).float()[None, None].to(matchers['SP+LG'].device) / 255.0
        with torch.no_grad():
            feats = matchers['SP+LG'].extractor.extract(t)
        return rbd(feats)['keypoints'].cpu().numpy()
    return None  # LoFTR: detector-free


def evaluate(pairs: list, matchers: dict) -> dict:
    """전체 쌍에 대해 각 알고리즘 평가."""
    results = {name: {'mma': [], 'auc': [], 'rep': []} for name in MATCHER_NAMES}

    for i, pair in enumerate(pairs):
        img0, img1, H = pair['img0'], pair['img1'], pair['H']

        for name, matcher in matchers.items():
            try:
                out = matcher.match(img0, img1)
                kpts0, kpts1, matches = out['kpts0'], out['kpts1'], out['matches']

                mma_scores = mma(kpts0, kpts1, matches, H, thresholds=(1, 3, 5))
                auc_scores = auc(kpts0, kpts1, matches, H, thresholds=(3, 5, 10))
                results[name]['mma'].append(mma_scores)
                results[name]['auc'].append(auc_scores)

                # Repeatability: LoFTR 제외
                if name != 'LoFTR':
                    k0 = detect_keypoints(name, img0, matchers)
                    k1 = detect_keypoints(name, img1, matchers)
                    rep = repeatability(k0, k1, H)
                    results[name]['rep'].append(rep)

            except Exception:
                results[name]['mma'].append({f'mma@{t}': 0.0 for t in (1, 3, 5)})
                results[name]['auc'].append({f'auc@{t}': 0.0 for t in (3, 5, 10)})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pairs)}]")

    return results


def print_results(results: dict):
    """결과 테이블 출력."""
    print("\n=== MMA (Mean Matching Accuracy) ===")
    print(f"{'Method':<10} {'MMA@1':>8} {'MMA@3':>8} {'MMA@5':>8}")
    print("-" * 38)
    for name in MATCHER_NAMES:
        scores = results[name]['mma']
        if not scores:
            continue
        m1 = np.mean([s['mma@1'] for s in scores])
        m3 = np.mean([s['mma@3'] for s in scores])
        m5 = np.mean([s['mma@5'] for s in scores])
        print(f"{name:<10} {m1:>8.3f} {m3:>8.3f} {m5:>8.3f}")

    print("\n=== AUC ===")
    print(f"{'Method':<10} {'AUC@3':>8} {'AUC@5':>8} {'AUC@10':>8}")
    print("-" * 38)
    for name in MATCHER_NAMES:
        scores = results[name]['auc']
        if not scores:
            continue
        a3  = np.mean([s['auc@3']  for s in scores])
        a5  = np.mean([s['auc@5']  for s in scores])
        a10 = np.mean([s['auc@10'] for s in scores])
        print(f"{name:<10} {a3:>8.3f} {a5:>8.3f} {a10:>8.3f}")

    print("\n=== Repeatability@3px (LoFTR 제외) ===")
    print(f"{'Method':<10} {'Rep@3':>8}")
    print("-" * 20)
    for name in MATCHER_NAMES:
        if name == 'LoFTR':
            print(f"{name:<10} {'N/A':>8}")
            continue
        reps = results[name]['rep']
        if reps:
            print(f"{name:<10} {np.mean(reps):>8.3f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pairs = load_hpatches('data/hpatches')
    print(f"Loaded {len(pairs)} pairs")

    matchers = {
        'SIFT':  SIFTMatcher(),
        'ORB':   ORBMatcher(),
        'SP+LG': SPLightGlueMatcher(device=device),
        'LoFTR': LoFTRMatcher(device=device),
    }

    results = evaluate(pairs, matchers)

    os.makedirs('results', exist_ok=True)

    # summary JSON for export_latex.py
    summary = {}
    for name in MATCHER_NAMES:
        mma_scores = results[name]['mma']
        auc_scores = results[name]['auc']
        rep_scores = results[name]['rep']
        summary[name] = {
            'mma@1':  float(np.mean([s['mma@1'] for s in mma_scores])) if mma_scores else 0.0,
            'mma@3':  float(np.mean([s['mma@3'] for s in mma_scores])) if mma_scores else 0.0,
            'mma@5':  float(np.mean([s['mma@5'] for s in mma_scores])) if mma_scores else 0.0,
            'auc@3':  float(np.mean([s['auc@3']  for s in auc_scores])) if auc_scores else 0.0,
            'auc@5':  float(np.mean([s['auc@5']  for s in auc_scores])) if auc_scores else 0.0,
            'auc@10': float(np.mean([s['auc@10'] for s in auc_scores])) if auc_scores else 0.0,
            'rep@3':  float(np.mean(rep_scores)) if rep_scores else None,
        }
    with open('results/matcher_eval.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print_results(results)
    print(f"\nSaved results/matcher_eval.json")


if __name__ == '__main__':
    main()
