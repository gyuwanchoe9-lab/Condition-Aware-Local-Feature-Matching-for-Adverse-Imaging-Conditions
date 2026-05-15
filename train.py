"""
PairClassifier 학습 스크립트.
labels/labels.json 생성 후 실행.
결과: model/weights.pth, figures/confusion_matrix.png
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from eval.hpatches import load_hpatches
from model.model import PairClassifier
from model.dataset import PairDataset, get_split

MATCHER_NAMES = ['SIFT', 'ORB', 'LoFTR', 'SP+LG']


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    한 에폭 실행.
    optimizer가 None이면 eval 모드 (역전파 없음).
    all_preds, all_labels 반환은 최종 평가 시에만 필요하므로 collect 옵션 제공.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img0, img1, label in loader:
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            if training:
                optimizer.zero_grad()
            out = model(img0, img1)
            loss = criterion(out, label)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(label)
            correct    += (out.argmax(1) == label).sum().item()
            total      += len(label)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_preds(model, loader, device):
    """eval set 전체 예측값과 정답 수집."""
    model.eval()
    all_preds, all_labels = [], []
    for img0, img1, label in loader:
        img0, img1 = img0.to(device), img1.to(device)
        preds = model(img0, img1).argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(label.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, n_classes: int = 4) -> dict:
    """Confusion matrix, per-class Precision / Recall / F1 계산."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    precision, recall, f1 = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    return {'confusion_matrix': cm, 'precision': precision, 'recall': recall, 'f1': f1}


def print_metrics(metrics: dict):
    """성능지표 출력."""
    print("\n=== Per-class Metrics ===")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for i, name in enumerate(MATCHER_NAMES):
        print(f"{name:<10} {metrics['precision'][i]:>10.3f} {metrics['recall'][i]:>10.3f} {metrics['f1'][i]:>10.3f}")
    macro_f1 = np.mean(metrics['f1'])
    print(f"\nMacro F1: {macro_f1:.3f}")


def save_confusion_matrix(cm: np.ndarray):
    """Confusion matrix를 figures/confusion_matrix.png 로 저장."""
    os.makedirs('figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(MATCHER_NAMES, rotation=15)
    ax.set_yticklabels(MATCHER_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved figures/confusion_matrix.png")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pairs = load_hpatches('data/hpatches')
    with open('labels/labels.json') as f:
        labels = json.load(f)

    train_samples, eval_samples = get_split(pairs, labels)
    print(f"Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    train_loader = DataLoader(PairDataset(train_samples), batch_size=16, shuffle=True,  num_workers=4)
    eval_loader  = DataLoader(PairDataset(eval_samples),  batch_size=16, shuffle=False, num_workers=4)

    model     = PairClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience, wait = 5, 0

    for epoch in range(50):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        ev_loss, ev_acc = run_epoch(model, eval_loader,  criterion, device)
        print(f"Epoch {epoch+1:2d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"eval  loss {ev_loss:.4f} acc {ev_acc:.3f}")

        if ev_loss < best_loss:
            best_loss = ev_loss
            wait = 0
            torch.save(model.state_dict(), 'model/weights.pth')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest eval loss: {best_loss:.4f}")

    # best 모델 로드 후 최종 성능 평가
    model.load_state_dict(torch.load('model/weights.pth'))
    preds, true_labels = collect_preds(model, eval_loader, device)

    metrics = compute_metrics(preds, true_labels)
    print_metrics(metrics)
    save_confusion_matrix(metrics['confusion_matrix'])


if __name__ == '__main__':
    main()
