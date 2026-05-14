import cv2
import numpy as np
import torch
import kornia.feature as KF
from lightglue import LightGlue
from lightglue import SuperPoint as LG_SuperPoint
from lightglue.utils import rbd


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _to_tensor(img_gray: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img_gray).float()[None, None].to(device) / 255.0


class SPLightGlueMatcher:
    """SuperPoint + LightGlue."""

    def __init__(self, device: str = 'cuda', n_features: int = 2048):
        self.device = device
        self.extractor = LG_SuperPoint(max_num_keypoints=n_features).eval().to(device)
        self.matcher   = LightGlue(features='superpoint').eval().to(device)

    @torch.no_grad()
    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        """SP+LG 매칭. 반환: kpts0, kpts1, matches."""
        f0 = self.extractor.extract(_to_tensor(_to_gray(img0), self.device))
        f1 = self.extractor.extract(_to_tensor(_to_gray(img1), self.device))
        out = self.matcher({'image0': f0, 'image1': f1})

        f0, f1, out = rbd(f0), rbd(f1), rbd(out)
        return {
            'kpts0': f0['keypoints'].cpu().numpy(),
            'kpts1': f1['keypoints'].cpu().numpy(),
            'matches': out['matches'].cpu().numpy(),
        }


class LoFTRMatcher:
    """LoFTR detector-free matcher (kornia)."""

    def __init__(self, device: str = 'cuda', pretrained: str = 'outdoor'):
        self.device = device
        self.loftr = KF.LoFTR(pretrained=pretrained).eval().to(device)

    @staticmethod
    def _resize(img: np.ndarray, max_side: int = 640) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
        return img[:h - h % 8, :w - w % 8]

    @torch.no_grad()
    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        """LoFTR 매칭. 반환: kpts0, kpts1, matches."""
        img0 = self._resize(_to_gray(img0))
        img1 = self._resize(_to_gray(img1))
        out = self.loftr({'image0': _to_tensor(img0, self.device),
                          'image1': _to_tensor(img1, self.device)})
        kpts0 = out['keypoints0'].cpu().numpy()
        n = len(kpts0)
        return {
            'kpts0': kpts0,
            'kpts1': out['keypoints1'].cpu().numpy(),
            'matches': np.stack([np.arange(n), np.arange(n)], axis=1),
        }
