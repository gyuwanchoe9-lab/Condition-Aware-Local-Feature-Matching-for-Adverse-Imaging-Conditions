import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ── 증강 함수 (generate_labels.py와 동일) ────────────────────────────────────

def aug_brightness(img: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.2, 0.7)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def aug_motion_blur(img: np.ndarray) -> np.ndarray:
    size = random.choice([9, 15, 21])
    angle = random.uniform(0, 180)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def aug_noise(img: np.ndarray) -> np.ndarray:
    sigma = random.uniform(20, 60)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


AUG_FN = {'brightness': aug_brightness, 'blur': aug_motion_blur, 'noise': aug_noise}


# ── Split ─────────────────────────────────────────────────────────────────────

def get_split(pairs: list, label_records: list, train_ratio: float = 0.8):
    """
    시퀀스 단위로 train/eval split.
    같은 시퀀스의 모든 증강 샘플이 같은 split에 들어가도록 leakage 방지.
    """
    scenes = sorted(set(p['scene'] for p in pairs))
    train_scenes = set(scenes[:int(len(scenes) * train_ratio)])

    # pair_id → raw pair 매핑
    pair_map = {p['pair_id']: p for p in pairs}

    train, eval_ = [], []
    for rec in label_records:
        pair = pair_map.get(rec['pair_id'])
        if pair is None:
            continue
        bucket = train if rec['scene'] in train_scenes else eval_
        bucket.append((pair, rec))

    return train, eval_


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """HPatches 이미지 쌍에 증강 적용 후 GT 라벨 반환."""

    _transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair, rec = self.samples[idx]
        aug_fn = AUG_FN[rec['aug']]

        # 라벨 생성 시와 동일한 증강 타입 적용 (강도는 랜덤)
        img0 = aug_fn(pair['img0'])
        img1 = aug_fn(pair['img1'])

        img0 = self._transform(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        img1 = self._transform(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        return img0, img1, torch.tensor(rec['label'], dtype=torch.long)
