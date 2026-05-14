"""
raw 이미지에서 brightness / blur / noise 수치를 추출하고
각 조건 구간별 알고리즘 선택 분포를 bar chart로 저장.
실행 전 labels/labels.json 이 있어야 함.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from eval.hpatches import load_hpatches
from iqa import measure_conditions

MATCHER_NAMES = ['SIFT', 'ORB', 'LoFTR', 'SP+LG']


def plot_distribution(conditions: list, labels: list, cond_name: str, n_bins: int = 5):
    """
    조건 수치를 n_bins 구간으로 나눠
    각 구간에서 알고리즘 선택 비율을 bar chart로 저장.
    """
    vals = np.array([c[cond_name] for c in conditions])
    labels_arr = np.array(labels)
    edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))

    x = np.arange(n_bins)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4))

    for m_idx, name in enumerate(MATCHER_NAMES):
        ratios = []
        for b in range(n_bins):
            mask = (vals >= edges[b]) & (vals <= edges[b + 1])
            ratios.append((labels_arr[mask] == m_idx).mean() if mask.sum() > 0 else 0.0)
        ax.bar(x + m_idx * width, ratios, width, label=name)

    bin_labels = [f'{edges[b]:.2f}~{edges[b+1]:.2f}' for b in range(n_bins)]
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(bin_labels, rotation=15, fontsize=8)
    ax.set_xlabel(cond_name)
    ax.set_ylabel('Selection ratio')
    ax.set_title(f'Algorithm selection by {cond_name}')
    ax.legend()
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    out_path = f'figures/dist_{cond_name}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    pairs = load_hpatches('data/hpatches')
    with open('labels/labels.json') as f:
        label_recs = json.load(f)

    label_map = {r['pair_id']: r['label'] for r in label_recs}

    conditions, labels = [], []
    for pair in pairs:
        pid = pair['pair_id']
        if pid not in label_map:
            continue
        # img0 기준으로 이미지 조건 측정
        conditions.append(measure_conditions(pair['img0']))
        labels.append(label_map[pid])

    # 조건별 분포 플롯
    for cond in ['brightness', 'blur', 'noise']:
        plot_distribution(conditions, labels, cond)

    # 전체 선택 분포 출력
    labels_arr = np.array(labels)
    print("\n=== Overall Algorithm Selection ===")
    for i, name in enumerate(MATCHER_NAMES):
        print(f"  {name}: {(labels_arr == i).mean()*100:.1f}%")


if __name__ == '__main__':
    main()
