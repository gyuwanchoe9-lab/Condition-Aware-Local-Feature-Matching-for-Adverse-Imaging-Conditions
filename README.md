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
   3가지 증강 각각 적용 (brightness / motion blur / gaussian noise)
   - Brightness: 픽셀값 × 0.2~0.7 (랜덤 어둡게)
   - Motion Blur: 랜덤 방향/크기 선형 커널 convolution
   - Gaussian Noise: σ=20~60 정규분포 노이즈 추가
         │
         ▼
   4개 알고리즘 각각 매칭 → MMA@3px → normalize → argmax
         │
         ▼
   GT 라벨 (0=SIFT / 1=ORB / 2=LoFTR / 3=SP+LG)
   + 조건 수치 (brightness, blur, noise 값) 저장
   → 총 580 × 3 = 1,740 샘플

2. Classification (80/20 split, 시퀀스 단위 랜덤 shuffle)
   증강된 이미지 쌍 (img0, img1)
         │
         ▼
   ResNet18 기반 PairClassifier (siamese 구조)
   - 클래스 불균형 보정: CrossEntropyLoss에 class weight 적용
   - Early stopping (patience=5, eval loss 기준)
         │
         ▼
   성능 지표:
   - Loss / Accuracy (에폭별)
   - Per-class Precision / Recall / F1
   - Macro F1
   - Confusion Matrix

3. Condition Analysis
   각 증강 조건 수치 구간별
   알고리즘 선택 분포 시각화 (bar chart × 3)
```

---

## Key Findings

- **Clean 이미지**: DL 방식(SP+LG, LoFTR)이 압도적으로 우세
- **Motion Blur**: LoFTR이 가장 강함 (detector-free 구조 덕분). SIFT도 25% 수준으로 선전
- **Brightness**: 조건에 상관없이 SP+LG, LoFTR 우세. 밝기가 알고리즘 선택에 미치는 영향 미미
- **Gaussian Noise**: SP+LG가 일관되게 강함 (60% 이상)
- **ORB**: 어떤 조건에서도 최적 알고리즘으로 거의 선택되지 않음 (3~5%)

---

## Data

- **HPatches**: 116 scenes × 5 pairs = **580 pairs** (GT homography 포함)
- 시퀀스 단위 랜덤 80/20 split (train: 93 scenes / eval: 23 scenes)

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
# 1. GT 라벨 생성 (580쌍 × 3증강 × 4알고리즘, 오래 걸림)
python generate_labels.py

# 2. 분류 모델 학습 + 성능 평가
python train.py

# 3. 조건별 분포 분석 및 시각화
python analysis.py

# 4. LaTeX 테이블 생성 (results/latex_tables.tex)
python export_latex.py
```

---

## Project Structure

```text
.
├── generate_labels.py   # GT 라벨 생성 (증강 포함)
├── train.py             # PairClassifier 학습 + 성능 평가
├── analysis.py          # 조건별 알고리즘 선택 분포 분석
├── export_latex.py      # 성능 지표 LaTeX 테이블 생성
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
└── figures/             # confusion matrix, 조건별 분포 bar chart
```

---

## Evaluation Metrics

### Loss (CrossEntropy) ↓ 낮을수록 좋음

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p(y_i)$$

- $p(y_i)$: 정답 클래스에 대한 예측 확률.

### Accuracy ↑ 높을수록 좋음

$$\text{Accuracy} = \frac{TP + TN}{N}$$

- 전체 샘플 중 맞춘 비율. 클래스 불균형 시 신뢰도 낮음.

### Precision ↑ 높을수록 좋음 (클래스 c)

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

- $TP_c$: c라고 예측 & 실제 c / $FP_c$: c라고 예측했지만 실제론 c가 아님
- 예측의 신뢰도

### Recall ↑ 높을수록 좋음 (클래스 c)

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

- $FN_c$: 실제 c인데 다른 클래스로 예측
- 실제 케이스를 얼마나 놓치지 않았는지

### F1 ↑ 높을수록 좋음 (클래스 c)

$$F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

- Precision과 Recall의 조화평균. 불균형 데이터에서 Accuracy 대신 사용.

### Macro F1 ↑ 높을수록 좋음

$$\text{Macro F1} = \frac{1}{C}\sum_{c=1}^{C} F1_c$$

- 클래스별 F1의 단순 평균. 샘플이 적은 클래스도 동등하게 반영.

### Confusion Matrix

- 행: 실제 라벨 / 열: 예측 라벨
- 대각선 값이 클수록 좋음 (올바르게 예측한 수)
- 어떤 알고리즘을 어떤 알고리즘으로 혼동하는지 시각화.

---

## References

- DeTone et al., *SuperPoint*, CVPRW 2018
- Sun et al., *LoFTR*, CVPR 2021
- Lindenberger et al., *LightGlue*, ICCV 2023
- Balntas et al., *HPatches*, CVPR 2017
