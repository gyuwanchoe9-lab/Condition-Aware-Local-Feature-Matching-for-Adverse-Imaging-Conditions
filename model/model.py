import torch
import torch.nn as nn
import torchvision.models as models


class PairClassifier(nn.Module):
    """
    이미지 쌍 (img0, img1)을 받아 최적 매칭 알고리즘 인덱스(0~3)를 예측.
    각 이미지를 공유 ResNet18 인코더로 특징 추출 후 concat → FC 4클래스.
    label: 0=SIFT, 1=ORB, 2=LoFTR, 3=SP+LG
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        # FC 레이어 제거, (B, 512, 1, 1) 출력
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        # img0, img1: (B, 3, H, W)
        f0 = self.encoder(img0).flatten(1)   # (B, 512)
        f1 = self.encoder(img1).flatten(1)   # (B, 512)
        return self.classifier(torch.cat([f0, f1], dim=1))  # (B, 4)
