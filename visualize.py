"""
논문용 매칭 결과 시각화.

사용법:
    conda run -n research python visualize.py              # 전체 figure 생성
    conda run -n research python visualize.py --scene i_ajuntament --pair 2
"""
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from augment import augment
from eval.hpatches import load_hpatches
from eval.metrics import _reproject_error

DEVICE = 'cuda'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'hpatches')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'figures')

CONDITIONS  = ['normal', 'low_light', 'motion_blur', 'combined']
COND_LABELS = ['Normal', 'Low-light', 'Motion-blur', 'Combined']
SEVERITIES  = {'normal': 0, 'low_light': 2, 'motion_blur': 2, 'combined': 2}


def draw_matches(img0, img1, kpts0, kpts1, matches, H,
                 inlier_thresh=3.0, max_draw=80) -> np.ndarray:
    """매칭쌍을 이미지 위에 그려서 side-by-side로 반환."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H_canvas = max(h0, h1)
    canvas = np.zeros((H_canvas, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0 if img0.ndim == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    canvas[:h1, w0:] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    if len(matches) == 0:
        return canvas

    m0 = kpts0[matches[:, 0]]
    m1 = kpts1[matches[:, 1]]
    errs = _reproject_error(m0, m1, H)
    inlier = errs < inlier_thresh

    # 너무 많으면 샘플링
    idx = np.where(inlier)[0]
    if len(idx) > max_draw:
        idx = np.random.choice(idx, max_draw, replace=False)
    idx_out = np.where(~inlier)[0]
    if len(idx_out) > max_draw // 4:
        idx_out = np.random.choice(idx_out, max_draw // 4, replace=False)

    for i in idx_out:
        p0 = (int(m0[i, 0]), int(m0[i, 1]))
        p1 = (int(m1[i, 0]) + w0, int(m1[i, 1]))
        cv2.line(canvas, p0, p1, (0, 0, 200), 1, cv2.LINE_AA)

    for i in idx:
        p0 = (int(m0[i, 0]), int(m0[i, 1]))
        p1 = (int(m1[i, 0]) + w0, int(m1[i, 1]))
        cv2.line(canvas, p0, p1, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, (0, 220, 0), -1)
        cv2.circle(canvas, p1, 2, (0, 220, 0), -1)

    n_in  = int(inlier.sum())
    n_all = len(matches)
    cv2.putText(canvas, f'{n_in}/{n_all}', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def get_matchers():
    from matchers.classical import SIFTMatcher, ORBMatcher
    from matchers.learned  import SuperPointMatcher, SPLightGlueMatcher, LoFTRMatcher
    from matchers.hybrid   import HybridMatcher

    sp = SuperPointMatcher(device=DEVICE)
    return {
        'SIFT':       SIFTMatcher(),
        'ORB':        ORBMatcher(),
        'SP+NN':      sp,
        'SP+LG':      SPLightGlueMatcher(device=DEVICE),
        'LoFTR':      LoFTRMatcher(device=DEVICE),
        'Hybrid':     HybridMatcher(device=DEVICE),
    }


def make_figure_methods_x_conditions(pair: dict, matchers: dict, save_path: str):
    """행=메서드, 열=조건 격자 figure."""
    n_methods = len(matchers)
    n_conds   = len(CONDITIONS)

    fig, axes = plt.subplots(n_methods, n_conds,
                             figsize=(5 * n_conds, 2.5 * n_methods))
    fig.suptitle(f"Scene: {pair['scene']}  |  Green=inlier  Red=outlier",
                 fontsize=12, y=1.01)

    for row, (name, matcher) in enumerate(matchers.items()):
        for col, cond in enumerate(CONDITIONS):
            sev = SEVERITIES[cond]
            img0 = augment(pair['img0'], cond, sev) if cond != 'normal' else pair['img0'].copy()
            img1 = augment(pair['img1'], cond, sev) if cond != 'normal' else pair['img1'].copy()

            try:
                out = matcher.match(img0, img1)
                canvas = draw_matches(img0, img1,
                                      out['kpts0'], out['kpts1'], out['matches'], pair['H'])
            except Exception:
                h, w = pair['img0'].shape[:2]
                canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)

            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            if row == 0:
                ax.set_title(COND_LABELS[col], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight='bold', rotation=90,
                              labelpad=4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def make_figure_severity(pair: dict, matcher, matcher_name: str,
                         condition: str, save_path: str):
    """severity별 매칭 변화 figure (1행 4열: normal + sev1/2/3)."""
    cases = [('Normal', pair['img0'].copy(), pair['img1'].copy())]
    for sev in [1, 2, 3]:
        cases.append((f'{condition} sev={sev}',
                      augment(pair['img0'], condition, sev),
                      augment(pair['img1'], condition, sev)))

    fig, axes = plt.subplots(1, 4, figsize=(20, 3))
    fig.suptitle(f'{matcher_name} | {condition} severity progression', fontsize=11)

    for ax, (label, img0, img1) in zip(axes, cases):
        try:
            out = matcher.match(img0, img1)
            canvas = draw_matches(img0, img1,
                                  out['kpts0'], out['kpts1'], out['matches'], pair['H'])
        except Exception:
            h, w = img0.shape[:2]
            canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=None, help='특정 scene 이름 (예: i_ajuntament)')
    parser.add_argument('--pair',  type=int, default=2, help='pair index 1-5 (기본: 2)')
    parser.add_argument('--fig',   choices=['grid', 'severity', 'all'], default='all')
    args = parser.parse_args()

    print("Loading HPatches...")
    pairs = load_hpatches(DATA_DIR)

    if args.scene:
        pairs = [p for p in pairs if p['scene'] == args.scene
                 and p['pair_id'].endswith(f'_1_{args.pair}')]
        if not pairs:
            print(f"Scene '{args.scene}' not found.")
            return
        pair = pairs[0]
    else:
        # 기본: 첫 번째 illumination scene
        pair = next(p for p in pairs if p['scene'].startswith('i_')
                    and p['pair_id'].endswith('_1_2'))

    print(f"Using pair: {pair['pair_id']}")
    matchers = get_matchers()

    if args.fig in ('grid', 'all'):
        make_figure_methods_x_conditions(
            pair, matchers,
            save_path=os.path.join(OUT_DIR, f"{pair['scene']}_grid.png"))

    if args.fig in ('severity', 'all'):
        sp_lg = matchers['SP+LG']
        for cond in ['low_light', 'motion_blur']:
            make_figure_severity(
                pair, sp_lg, 'SP+LG', cond,
                save_path=os.path.join(OUT_DIR, f"{pair['scene']}_severity_{cond}.png"))


if __name__ == '__main__':
    main()
