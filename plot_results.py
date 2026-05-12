"""
CSV → 논문용 figure 자동 생성.
    conda run -n research python plot_results.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 색상/스타일 ────────────────────────────────────────────────────────────────
METHOD_STYLE = {
    'sift':              {'color': '#4878CF', 'marker': 'o',  'ls': '--',  'label': 'SIFT'},
    'orb':               {'color': '#6ACC65', 'marker': 's',  'ls': '--',  'label': 'ORB'},
    'sp+nn':             {'color': '#D65F5F', 'marker': '^',  'ls': '-',   'label': 'SP+NN'},
    'sp+lg':             {'color': '#B47CC7', 'marker': 'D',  'ls': '-',   'label': 'SP+LG'},
    'loftr':             {'color': '#C4AD66', 'marker': 'v',  'ls': '-',   'label': 'LoFTR'},
    'hybrid':            {'color': '#77BEDB', 'marker': '*',  'ls': '-',   'label': 'Hybrid (Ours)', 'lw': 2},
    'sp_det+sift_desc':  {'color': '#FF9500', 'marker': 'x',  'ls': ':',   'label': 'SP det + SIFT desc'},
    'sift_det+sp_desc':  {'color': '#FF2D55', 'marker': '+',  'ls': ':',   'label': 'SIFT det + SP desc'},
    'sp_det+sp_desc':    {'color': '#D65F5F', 'marker': '^',  'ls': ':',   'label': 'SP det + SP desc'},
    'sift_det+sift_desc':{'color': '#4878CF', 'marker': 'o',  'ls': ':',   'label': 'SIFT det + SIFT desc'},
}

COND_LABEL = {
    'normal':       'Normal',
    'low_light':    'Low-light',
    'motion_blur':  'Motion-blur',
    'combined':     'Combined',
}

MAIN_METHODS   = ['sift', 'orb', 'sp+nn', 'sp+lg', 'loftr', 'hybrid']
CROSS_METHODS  = ['sp_det+sp_desc', 'sp_det+sift_desc',
                  'sift_det+sp_desc', 'sift_det+sift_desc']
CONDITIONS     = ['normal', 'low_light', 'motion_blur', 'combined']


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['method'] = df['method'].str.lower().str.strip()
    return df


# ── Figure 1: 조건별 MAA@3 bar chart (Table B 대응) ──────────────────────────
def plot_table_b(df: pd.DataFrame):
    """메서드 × 조건, severity 평균 MAA@3."""
    methods = [m for m in MAIN_METHODS if m in df['method'].unique()]
    conds   = [c for c in CONDITIONS if c in df['condition'].unique()]

    # severity 평균 (normal은 severity=0 하나뿐)
    agg = df[df['method'].isin(methods)].groupby(['method', 'condition'])['maa@3'].mean()

    x     = np.arange(len(conds))
    width = 0.12
    fig, ax = plt.subplots(figsize=(7, 3.5))

    for i, method in enumerate(methods):
        vals = [agg.get((method, c), 0) for c in conds]
        st   = METHOD_STYLE.get(method, {})
        bars = ax.bar(x + i * width, vals, width,
                      label=st.get('label', method),
                      color=st.get('color', 'gray'),
                      edgecolor='white', linewidth=0.5)
        # Hybrid 강조
        if method == 'hybrid':
            for bar in bars:
                bar.set_edgecolor('#333')
                bar.set_linewidth(1.2)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([COND_LABEL[c] for c in conds])
    ax.set_ylabel('MAA @ 3px')
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(ncol=3, frameon=False, loc='upper right')
    ax.set_title('Full Pipeline MAA@3px by Condition (severity avg.)')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'table_b_bar.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: Severity degradation curve ─────────────────────────────────────
def plot_severity_curves(df: pd.DataFrame):
    """조건별 severity 증가에 따른 MAA@3 하락 곡선."""
    methods = [m for m in MAIN_METHODS if m in df['method'].unique()]
    deg_conds = [c for c in ['low_light', 'motion_blur', 'combined'] if c in df['condition'].unique()]

    fig, axes = plt.subplots(1, len(deg_conds), figsize=(4 * len(deg_conds), 3.2), sharey=True)
    if len(deg_conds) == 1:
        axes = [axes]

    for ax, cond in zip(axes, deg_conds):
        sub = df[(df['condition'] == cond) & (df['method'].isin(methods))]
        for method in methods:
            m_df = sub[sub['method'] == method].sort_values('severity')
            if m_df.empty:
                continue
            st = METHOD_STYLE.get(method, {})
            lw = st.get('lw', 1.2)
            ax.plot(m_df['severity'], m_df['maa@3'],
                    color=st.get('color', 'gray'),
                    marker=st.get('marker', 'o'),
                    ls=st.get('ls', '-'),
                    linewidth=lw,
                    markersize=4,
                    label=st.get('label', method))
        ax.set_title(COND_LABEL[cond])
        ax.set_xlabel('Severity')
        ax.set_xticks([1, 2, 3])
        ax.set_ylim(0, 0.85)

    axes[0].set_ylabel('MAA @ 3px')

    # 공통 legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.12))

    fig.suptitle('Performance Degradation vs. Severity', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'severity_curves.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Cross-combination heatmap (Table C 대응) ───────────────────────
def plot_cross_heatmap(df: pd.DataFrame):
    """Detector × Descriptor 분해 분석 heatmap."""
    cross = [m for m in CROSS_METHODS if m in df['method'].unique()]
    conds = [c for c in ['normal', 'low_light', 'motion_blur'] if c in df['condition'].unique()]
    if not cross:
        print("Cross-combination data not found, skipping.")
        return

    agg = df[df['method'].isin(cross)].groupby(['method', 'condition'])['maa@3'].mean()
    data = np.array([[agg.get((m, c), 0) for c in conds] for m in cross])

    row_labels = [METHOD_STYLE.get(m, {}).get('label', m) for m in cross]
    col_labels  = [COND_LABEL[c] for c in conds]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=0.5, aspect='auto')

    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)

    for i in range(len(cross)):
        for j in range(len(conds)):
            ax.text(j, i, f'{data[i, j]:.3f}',
                    ha='center', va='center', fontsize=8,
                    color='black' if data[i, j] > 0.15 else 'white')

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='MAA@3px')
    ax.set_title('Cross-combination: Detector vs Descriptor (Table C)')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'table_c_heatmap.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── LaTeX 표 자동 생성 ────────────────────────────────────────────────────────
def export_latex_table_b(df: pd.DataFrame):
    """Overleaf에서 바로 컴파일되는 standalone LaTeX 문서."""
    methods = [m for m in MAIN_METHODS if m in df['method'].unique()]
    conds   = [c for c in CONDITIONS if c in df['condition'].unique()]
    agg     = df[df['method'].isin(methods)].groupby(['method', 'condition'])['maa@3'].mean()

    lines = []
    # 완전한 문서 구조
    lines.append(r'\documentclass{article}')
    lines.append(r'\usepackage{booktabs}')
    lines.append(r'\usepackage{colortbl}')
    lines.append(r'\usepackage[table]{xcolor}')
    lines.append(r'\usepackage[margin=1in]{geometry}')
    lines.append(r'\begin{document}')
    lines.append(r'')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{Full Pipeline MAA@3px by Condition (avg. over severity)}')
    lines.append(r'\begin{tabular}{l' + 'c' * len(conds) + '}')
    lines.append(r'\toprule')
    lines.append('Method & ' + ' & '.join(COND_LABEL[c] for c in conds) + r' \\')
    lines.append(r'\midrule')

    for method in methods:
        label = METHOD_STYLE.get(method, {}).get('label', method)
        vals  = [agg.get((method, c), 0) for c in conds]
        row = label
        for c, v in zip(conds, vals):
            col_best = max(agg.get((m, c), 0) for m in methods)
            cell = f'\\textbf{{{v:.3f}}}' if abs(v - col_best) < 1e-4 else f'{v:.3f}'
            row += f' & {cell}'
        row += r' \\'
        if method == 'hybrid':
            row = r'\rowcolor{blue!8} ' + row
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append(r'')
    lines.append(r'\end{document}')

    path = os.path.join(OUT_DIR, 'table_b.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'results', 'pipeline_results.csv')
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        print("먼저 run_experiments.py 실행 필요.")
        return

    df = load(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    plot_table_b(df)
    plot_severity_curves(df)
    plot_cross_heatmap(df)
    export_latex_table_b(df)

    print(f"\n모든 figure → {OUT_DIR}/")


if __name__ == '__main__':
    main()
