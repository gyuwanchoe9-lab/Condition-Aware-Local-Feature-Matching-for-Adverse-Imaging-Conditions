import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_split(pairs: list, labels: list, train_ratio: float = 0.8):
    """
    시퀀스 단위로 train/eval split.
    같은 시퀀스 내 쌍이 양쪽에 섞이지 않도록 leakage 방지.
    """
    scenes = sorted(set(p['scene'] for p in pairs))
    train_scenes = set(scenes[:int(len(scenes) * train_ratio)])

    label_map = {r['pair_id']: r for r in labels}

    train, eval_ = [], []
    for p in pairs:
        if p['pair_id'] not in label_map:
            continue
        bucket = train if p['scene'] in train_scenes else eval_
        bucket.append((p, label_map[p['pair_id']]))

    return train, eval_


class PairDataset(Dataset):
    """HPatches 이미지 쌍과 GT 라벨을 반환하는 Dataset."""

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
        # BGR → RGB 변환 후 transform 적용
        img0 = self._transform(cv2.cvtColor(pair['img0'], cv2.COLOR_BGR2RGB))
        img1 = self._transform(cv2.cvtColor(pair['img1'], cv2.COLOR_BGR2RGB))
        label = torch.tensor(rec['label'], dtype=torch.long)
        return img0, img1, label
