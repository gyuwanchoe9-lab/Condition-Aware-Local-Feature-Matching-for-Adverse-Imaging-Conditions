"""
학습된 모델로 eval set 추론 후 LaTeX 테이블 코드 출력.
실행: python export_latex.py
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.hpatches import load_hpatches
from model.model import PairClassifier
from model.dataset import PairDataset, get_split
from train import collect_preds, compute_metrics

MATCHER_NAMES = ['SIFT', 'ORB', 'LoFTR', 'SP+LG']


def print_metrics_table(metrics: dict, acc: float):
    """Per-class Precision / Recall / F1 LaTeX 테이블 출력."""
    print("% ── Per-class Metrics Table ──────────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Per-class Classification Metrics on Eval Set}")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"\textbf{Algorithm} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\")
    print(r"\hline")
    for i, name in enumerate(MATCHER_NAMES):
        p = metrics['precision'][i]
        r = metrics['recall'][i]
        f = metrics['f1'][i]
        print(f"{name} & {p:.3f} & {r:.3f} & {f:.3f} \\\\")
    print(r"\hline")
    macro_f1 = np.mean(metrics['f1'])
    print(f"\\textbf{{Macro F1}} & & & {macro_f1:.3f} \\\\")
    print(f"\\textbf{{Accuracy}} & & & {acc:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def print_confusion_matrix_table(cm: np.ndarray):
    """Confusion Matrix LaTeX 테이블 출력."""
    print("% ── Confusion Matrix Table ───────────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Confusion Matrix (Rows: True, Columns: Predicted)}")
    header = " & " + " & ".join([f"\\textbf{{{n}}}" for n in MATCHER_NAMES]) + r" \\"
    print(r"\begin{tabular}{l" + "c" * 4 + "}")
    print(r"\hline")
    print(r"\textbf{True \textbackslash Pred}" + header)
    print(r"\hline")
    for i, name in enumerate(MATCHER_NAMES):
        row = name + " & " + " & ".join(str(cm[i, j]) for j in range(4)) + r" \\"
        print(row)
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def print_distribution_table(label_records: list):
    """증강 조건별 알고리즘 선택 비율 LaTeX 테이블 출력."""
    print("% ── Algorithm Selection Distribution Table ───────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Algorithm Selection Ratio by Augmentation Condition}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"\textbf{Condition} & \textbf{SIFT} & \textbf{ORB} & \textbf{LoFTR} & \textbf{SP+LG} \\")
    print(r"\hline")
    for aug in ['brightness', 'blur', 'noise']:
        subset = [r for r in label_records if r['aug'] == aug]
        labels = np.array([r['label'] for r in subset])
        ratios = [(labels == i).mean() for i in range(4)]
        row = aug.capitalize() + " & " + " & ".join(f"{r:.3f}" for r in ratios) + r" \\"
        print(row)
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pairs = load_hpatches('data/hpatches')
    with open('labels/labels.json') as f:
        label_records = json.load(f)

    _, eval_samples = get_split(pairs, label_records)
    eval_loader = DataLoader(PairDataset(eval_samples), batch_size=16, shuffle=False, num_workers=4)

    model = PairClassifier().to(device)
    model.load_state_dict(torch.load('model/weights.pth', map_location=device))

    preds, true_labels = collect_preds(model, eval_loader, device)
    acc = (preds == true_labels).mean()
    metrics = compute_metrics(preds, true_labels)

    import os, sys
    os.makedirs('results', exist_ok=True)
    out_path = 'results/latex_tables.tex'

    with open(out_path, 'w') as f:
        sys.stdout = f
        print(r"\documentclass{article}")
        print(r"\usepackage{booktabs}")
        print(r"\begin{document}")
        print()
        print_metrics_table(metrics, acc)
        print_confusion_matrix_table(metrics['confusion_matrix'])
        print_distribution_table(label_records)
        print()
        print(r"\end{document}")
        sys.stdout = sys.__stdout__

    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
