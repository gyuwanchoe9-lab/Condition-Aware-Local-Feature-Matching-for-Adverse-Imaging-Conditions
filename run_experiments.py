"""
메인 실험 스크립트.
결과는 results/ 폴더에 CSV로 저장.

사용법:
    conda run -n research python run_experiments.py --level all
    conda run -n research python run_experiments.py --level detector
    conda run -n research python run_experiments.py --level pipeline --condition low_light
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from augment import augment, AUGMENTATIONS
from eval.hpatches import load_hpatches
from eval.run_eval import eval_detector, eval_pipeline

DEVICE = 'cuda'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'hpatches')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEVERITIES = [1, 2, 3]
CONDITIONS = ['normal', 'low_light', 'motion_blur', 'gaussian_noise', 'combined']


def load_matchers(level: str):
    from matchers.classical import SIFTMatcher, ORBMatcher, CrossCombinationMatcher
    from matchers.learned import SuperPointMatcher, SPLightGlueMatcher, LoFTRMatcher
    from matchers.hybrid import HybridMatcher

    sp = SuperPointMatcher(device=DEVICE)

    pipeline_matchers = {
        'sift':   SIFTMatcher(),
        'orb':    ORBMatcher(),
        'sp+nn':  sp,
        'sp+lg':  SPLightGlueMatcher(device=DEVICE),
        'loftr':  LoFTRMatcher(device=DEVICE),
        'hybrid': HybridMatcher(device=DEVICE),
    }

    cross_matchers = {
        'sp_det+sift_desc': CrossCombinationMatcher('sp', 'sift', sp_matcher=sp),
        'sift_det+sp_desc': CrossCombinationMatcher('sift', 'sp', sp_matcher=sp),
        'sp_det+sp_desc':   CrossCombinationMatcher('sp', 'sp', sp_matcher=sp),
        'sift_det+sift_desc': CrossCombinationMatcher('sift', 'sift'),
    }

    detector_matchers = {
        'dog(sift)': SIFTMatcher(),
        'fast(orb)': ORBMatcher(),
        'sp_det':    sp,
    }

    if level == 'detector':
        return detector_matchers, {}
    elif level == 'pipeline':
        return {}, {**pipeline_matchers, **cross_matchers}
    else:  # all
        return detector_matchers, {**pipeline_matchers, **cross_matchers}


def augment_pairs(pairs, condition: str, severity: int):
    out = []
    for p in pairs:
        out.append({
            **p,
            'img0': augment(p['img0'], condition, severity),
            'img1': augment(p['img1'], condition, severity),
        })
    return out


def run_detector_table(detector_matchers, pairs_base):
    rows = []
    for name, matcher in detector_matchers.items():
        for condition in ['normal', 'low_light', 'motion_blur', 'combined']:
            for sev in ([0] if condition == 'normal' else SEVERITIES):
                if condition == 'normal':
                    pairs = pairs_base
                else:
                    pairs = augment_pairs(pairs_base, condition, sev)
                res = eval_detector(matcher, pairs)
                rows.append({'method': name, 'condition': condition, 'severity': sev, **res})
                print(f"[detector] {name} | {condition} sev={sev} | rep={res['repeatability']:.3f}")
    return pd.DataFrame(rows)


def run_pipeline_table(pipeline_matchers, pairs_base):
    rows = []
    for name, matcher in pipeline_matchers.items():
        for condition in ['normal', 'low_light', 'motion_blur', 'combined']:
            for sev in ([0] if condition == 'normal' else SEVERITIES):
                if condition == 'normal':
                    pairs = pairs_base
                else:
                    pairs = augment_pairs(pairs_base, condition, sev)
                res = eval_pipeline(matcher, pairs)
                rows.append({'method': name, 'condition': condition, 'severity': sev, **res})
                print(f"[pipeline] {name} | {condition} sev={sev} | maa@3={res['maa@3']:.3f}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', choices=['detector', 'pipeline', 'all'], default='all')
    parser.add_argument('--condition', default=None, help='단일 조건만 실행할 때 (optional)')
    parser.add_argument('--max_pairs', type=int, default=None, help='빠른 디버깅용 페어 수 제한')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading HPatches...")
    pairs_base = load_hpatches(DATA_DIR)
    if args.max_pairs:
        pairs_base = pairs_base[:args.max_pairs]
    print(f"  {len(pairs_base)} pairs loaded.")

    detector_matchers, pipeline_matchers = load_matchers(args.level)

    if args.condition:
        global CONDITIONS
        CONDITIONS = [args.condition]

    if detector_matchers:
        df_det = run_detector_table(detector_matchers, pairs_base)
        path = os.path.join(RESULTS_DIR, 'detector_results.csv')
        df_det.to_csv(path, index=False)
        print(f"\nSaved: {path}")

    if pipeline_matchers:
        df_pipe = run_pipeline_table(pipeline_matchers, pairs_base)
        path = os.path.join(RESULTS_DIR, 'pipeline_results.csv')
        df_pipe.to_csv(path, index=False)
        print(f"\nSaved: {path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
