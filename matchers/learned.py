import cv2
import numpy as np
import torch
import kornia.feature as KF
from lightglue import LightGlue
from lightglue import SuperPoint as LG_SuperPoint
from lightglue.utils import rbd


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _np_to_tensor(img_gray: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img_gray).float()[None, None].to(device) / 255.0


class SuperPointMatcher:
    """SuperPoint detector + descriptor, mutual NN matching."""

    def __init__(self, device: str = 'cuda', n_features: int = 2048):
        self.device = device
        self.extractor = LG_SuperPoint(max_num_keypoints=n_features).eval().to(device)

    @torch.no_grad()
    def _extract(self, img_gray: np.ndarray) -> dict:
        t = _np_to_tensor(img_gray, self.device)
        return self.extractor.extract(t)

    def detect(self, img_gray: np.ndarray) -> np.ndarray:
        feats = self._extract(_to_gray(img_gray))
        return feats['keypoints'][0].cpu().numpy()  # (N, 2)

    def describe(self, img_gray: np.ndarray, kpts: np.ndarray) -> np.ndarray:
        feats = self._extract(_to_gray(img_gray))
        return feats['descriptors'][0].cpu().numpy()  # (N, 256)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        f0 = self._extract(_to_gray(img0))
        f1 = self._extract(_to_gray(img1))

        kpts0 = f0['keypoints'][0].cpu().numpy()
        kpts1 = f1['keypoints'][0].cpu().numpy()
        d0 = torch.from_numpy(f0['descriptors'][0].cpu().numpy())
        d1 = torch.from_numpy(f1['descriptors'][0].cpu().numpy())

        if len(kpts0) == 0 or len(kpts1) == 0:
            return {'kpts0': kpts0, 'kpts1': kpts1, 'matches': np.empty((0, 2), dtype=int)}

        # L2 mutual NN
        sim = d0 @ d1.T
        nn01 = sim.argmax(dim=1)
        nn10 = sim.argmax(dim=0)
        ids0 = torch.arange(len(kpts0))
        mutual = nn10[nn01] == ids0
        matches = torch.stack([ids0[mutual], nn01[mutual]], dim=1).numpy()
        return {'kpts0': kpts0, 'kpts1': kpts1, 'matches': matches}


class SPLightGlueMatcher:
    """SuperPoint + LightGlue (ETH CVG)."""

    def __init__(self, device: str = 'cuda', n_features: int = 2048):
        self.device = device
        self.extractor = LG_SuperPoint(max_num_keypoints=n_features).eval().to(device)
        self.matcher   = LightGlue(features='superpoint').eval().to(device)

    @torch.no_grad()
    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        def prep(img):
            return _np_to_tensor(_to_gray(img), self.device)

        f0 = self.extractor.extract(prep(img0))
        f1 = self.extractor.extract(prep(img1))
        out = self.matcher({'image0': f0, 'image1': f1})

        f0, f1, out = rbd(f0), rbd(f1), rbd(out)
        kpts0 = f0['keypoints'].cpu().numpy()
        kpts1 = f1['keypoints'].cpu().numpy()
        matches = out['matches'].cpu().numpy()  # (M, 2)

        return {'kpts0': kpts0, 'kpts1': kpts1, 'matches': matches}


class LoFTRMatcher:
    """LoFTR detector-free matcher (kornia)."""

    def __init__(self, device: str = 'cuda', pretrained: str = 'outdoor'):
        self.device = device
        self.loftr = KF.LoFTR(pretrained=pretrained).eval().to(device)

    @torch.no_grad()
    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        t0 = _np_to_tensor(_to_gray(img0), self.device)
        t1 = _np_to_tensor(_to_gray(img1), self.device)
        out = self.loftr({'image0': t0, 'image1': t1})

        kpts0 = out['keypoints0'].cpu().numpy()
        kpts1 = out['keypoints1'].cpu().numpy()
        n = len(kpts0)
        matches = np.stack([np.arange(n), np.arange(n)], axis=1)
        return {'kpts0': kpts0, 'kpts1': kpts1, 'matches': matches}
