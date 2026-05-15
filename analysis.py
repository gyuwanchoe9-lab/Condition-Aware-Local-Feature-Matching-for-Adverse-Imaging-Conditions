"""
labels/labels.json에서 조건 수치를 읽어
각 증강 조건별 알고리즘 선택 분포를 bar chart로 저장.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

MATCHER_NAMES = ['SIFT', 'ORB', 'LoFTR', 'SP+LG']
AUG_NAMES = ['brightness', 'blur', 'noise']


def plot_distribution(records: list, aug_name: str, cond_name: str, n_bins: int = 5):
    """
    특정 증강 조건의 샘플만 필터링해
    조건 수치 구간별 알고리즘 선택 비율을 bar chart로 저장.
    """
    subset = [r for r in records if r['aug'] == aug_name]
    vals = np.array([r['conditions'][cond_name] for r in subset])
    labels = np.array([r['label'] for r in subset])
    edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))

    x = np.arange(n_bins)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4))

    for m_idx, name in enumerate(MATCHER_NAMES):
        ratios = []
        for b in range(n_bins):
            mask = (vals >= edges[b]) & (vals <= edges[b + 1])
            ratios.append((labels[mask] == m_idx).mean() if mask.sum() > 0 else 0.0)
        ax.bar(x + m_idx * width, ratios, width, label=name)

    bin_labels = [f'{edges[b]:.2f}~{edges[b+1]:.2f}' for b in range(n_bins)]
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(bin_labels, rotation=15, fontsize=8)
    ax.set_xlabel(cond_name)
    ax.set_ylabel('Selection ratio')
    ax.set_title(f'[{aug_name}] Algorithm selection by {cond_name}')
    ax.legend()
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    out_path = f'figures/dist_{aug_name}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    with open('labels/labels.json') as f:
        records = json.load(f)

    # 각 증강 조건별로 해당 조건 수치 기준 분포 플롯
    cond_map = {'brightness': 'brightness', 'blur': 'blur', 'noise': 'noise'}
    for aug_name, cond_name in cond_map.items():
        plot_distribution(records, aug_name, cond_name)

    # 증강 조건별 전체 알고리즘 선택 분포 출력
    print()
    for aug_name in AUG_NAMES:
        subset = [r for r in records if r['aug'] == aug_name]
        labels = np.array([r['label'] for r in subset])
        print(f"=== {aug_name} ===")
        for i, name in enumerate(MATCHER_NAMES):
            print(f"  {name}: {(labels == i).mean()*100:.1f}%")
        print()


if __name__ == '__main__':
    main()
