"""
PairClassifier 학습 스크립트.
labels/labels.json 생성 후 실행.
결과: model/weights.pth
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval.hpatches import load_hpatches
from model.model import PairClassifier
from model.dataset import PairDataset, get_split


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    한 에폭 실행.
    optimizer가 None이면 eval 모드 (역전파 없음).
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

        # eval loss 기준 best 모델 저장
        if ev_loss < best_loss:
            best_loss = ev_loss
            wait = 0
            torch.save(model.state_dict(), 'model/weights.pth')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best eval loss: {best_loss:.4f} | Saved model/weights.pth")


if __name__ == '__main__':
    main()
