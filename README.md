# Condition-Aware Local Feature Matching for Adverse Imaging Conditions

> 2026 컴퓨터비전 텀프로젝트 — 서울과학기술대학교 전기정보공학과

---

## 개요

HPatches 데이터셋을 기반으로 **4가지 특징점 매칭 알고리즘**의 성능을 비교하고, 이미지의 내재적 조건(밝기, 블러, 노이즈)에 따라 어떤 알고리즘이 최적인지를 분류·분석하는 프로젝트입니다.

> *하이브리드 매처처럼 알고리즘을 직접 혼합하는 것이 아니라, 조건별 알고리즘 선택 분포를 통해 "어떤 상황에서 어떤 알고리즘이 유리한가"를 분석하는 것이 목표입니다.*

---

## Algorithms

| 카테고리 | 알고리즘 |
| --- | --- |
| Classical | SIFT, ORB |
| Deep Learning | LoFTR, SuperPoint + LightGlue |

---

## Pipeline

```text
1. Label Generation
   HPatches 이미지 쌍
         │
         ▼
   4개 알고리즘 각각 매칭
         │
         ▼
   MMA@3px 스코어 → normalize → argmax
         │
         ▼
   GT 라벨 (0=SIFT / 1=ORB / 2=LoFTR / 3=SP+LG)

2. Classification
   이미지 쌍 (img0, img1)
         │
         ▼
   ResNet18 기반 PairClassifier
         │
         ▼
   예측 라벨 vs GT 라벨 → loss, accuracy

3. Condition Analysis
   이미지에서 brightness / blur / noise 수치 추출
         │
         ▼
   조건 구간별 알고리즘 선택 분포 시각화
```

---

## Data

- **HPatches**: 116 scenes × 5 pairs = **580 pairs** (GT homography 포함)
- 80/20 시퀀스 단위 split (train: 93 scenes / eval: 23 scenes)

```bash
cd data
wget https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
unzip hpatches-sequences-release.zip
```

---

## Setup

```bash
pip install torch torchvision kornia git+https://github.com/cvg/LightGlue.git matplotlib
```

---

## Run

```bash
# 1. GT 라벨 생성 (580쌍 × 4알고리즘, 오래 걸림)
python generate_labels.py

# 2. 분류 모델 학습
python train.py

# 3. 조건별 분포 분석 및 시각화
python analysis.py
```

---

## Project Structure

```text
.
├── generate_labels.py   # GT 라벨 생성
├── train.py             # PairClassifier 학습
├── analysis.py          # 조건별 알고리즘 선택 분포 분석
├── iqa.py               # brightness / blur / noise 수치 추출
├── matchers/
│   ├── classical.py     # SIFT, ORB
│   └── learned.py       # LoFTR, SP+LightGlue
├── eval/
│   ├── metrics.py       # MMA 계산
│   └── hpatches.py      # HPatches 데이터 로더
├── model/
│   ├── model.py         # PairClassifier (ResNet18 기반)
│   └── dataset.py       # 이미지 쌍 Dataset / split
├── labels/
│   └── labels.json      # 생성된 GT 라벨
└── figures/             # 분석 결과 시각화
```

---

## References

- DeTone et al., *SuperPoint*, CVPRW 2018
- Sun et al., *LoFTR*, CVPR 2021
- Lindenberger et al., *LightGlue*, ICCV 2023
- Balntas et al., *HPatches*, CVPR 2017
