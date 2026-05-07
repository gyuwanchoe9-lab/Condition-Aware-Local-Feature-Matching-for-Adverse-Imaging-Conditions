# Condition-Aware Local Feature Matching for Adverse Imaging Conditions

> 2026 컴퓨터비전 텀프로젝트 — 서울과학기술대학교 전기정보공학과

악조건(저조도, 모션블러, 노이즈)에서의 Local Feature Matching 비교 분석 및 Condition-Aware Hybrid Matching 제안.

---

## 1. Overview

핸드헬드 야간 촬영 페어 매칭 시나리오에서 SIFT, ORB, SuperPoint, LoFTR 등 주요 local feature matching 방법을 체계적으로 비교 분석하고, 입력 영상의 조건을 자동 판별하여 최적 method를 선택하는 hybrid matcher를 제안한다.

### 1.1 Motivation

SuperPoint 논문(DeTone et al., 2018)은 학습 시 motion blur, brightness, Gaussian noise를 augmentation으로 사용했음을 training details에서 명시하지만, **평가 단계에서는 이러한 악조건에 대한 정량 분석을 제공하지 않는다.** 본 프로젝트는 이 빈틈에 다음 세 질문으로 답한다:

1. Augmentation으로 학습된 robustness가 실제 악조건 평가에 얼마나 전이되는가?
2. Detector level과 descriptor level 중 어느 쪽이 먼저 무너지는가?
3. 그 분석을 바탕으로 condition-aware hybrid matching이 가능한가?

단순 SuperPoint 재현이 아닌, **언제 어떤 method를 써야 하는가**에 대한 분해 분석과 자동 선택 메커니즘을 제안한다.

### 1.2 Application Scenario

- 핸드헬드 폰 촬영의 야간 길거리 / 실내 저조도 환경
- 두 장 이미지 페어 간 호모그래피 추정
- Stitching, planar AR alignment, visual localization 전처리 등 응용 가능

---

## 2. Method Lineup

| 카테고리 | 모델 | 역할 |
| --- | --- | --- |
| Classical full pipeline | SIFT, ORB | classical baseline |
| Learned detector-only | SuperPoint (detector head only) | learned detector 단독 평가 |
| Learned full pipeline | SuperPoint + Mutual NN | SuperPoint 논문 원본 셋업 |
| Modern learned matcher | SuperPoint + LightGlue | 현재 SOTA 수준 |
| Detector-free | LoFTR | 다른 패러다임 비교 |
| Cross-combination | SP det + SIFT desc, SIFT det + SP desc | Detector vs Descriptor 분해 분석 |
| **Ours** | **Condition-aware Hybrid** | **본 프로젝트 기여** |

---

## 3. Evaluation Protocol

세 레벨로 분리하여 평가 (HPatches 프로토콜 + LightGlue식 MAA 기반).

### 3.1 Level 1 — Detector-only

| Metric | 설명 |
| --- | --- |
| Repeatability (ε=3) | 두 이미지에서 검출된 keypoint가 H로 warp 후 ε pixel 안에 있는 비율 |
| Coverage | Grid cell당 keypoint 수의 분산 (분포 균일성) |

### 3.2 Level 2 — Descriptor

| Metric | 설명 |
| --- | --- |
| MMA @ {1, 3, 5} | Threshold별 매칭 정확도 (D2-Net, R2D2 표준) |

### 3.3 Level 3 — Full Pipeline

| Metric | 설명 |
| --- | --- |
| MAA @ {1, 3, 5, 10} | Corner reprojection error 누적분포 AUC |
| Inlier Ratio | RANSAC 후 inlier 비율 |
| Runtime | Detection + Description + Matching (ms) |

---

## 4. Result Tables

### 표 A — Detector-only Repeatability (조건별)

| Detector | Normal | Low-light | Motion-blur | Low-light + Blur |
| --- | --- | --- | --- | --- |
| DoG (SIFT) | | | | |
| FAST (ORB) | | | | |
| SuperPoint (det head) | | | | |

### 표 B — Full Pipeline MAA @3px (조건별)

| Method | Normal | Low-light | Motion-blur | Low-light + Blur |
| --- | --- | --- | --- | --- |
| SIFT | | | | |
| ORB | | | | |
| SuperPoint + NN | | | | |
| SuperPoint + LightGlue | | | | |
| LoFTR | | | | |
| **Ours (Hybrid)** | | | | |

### 표 C — Cross-combination 분해 분석 (MMA @3px)

| Detector | Descriptor | Normal | Low-light | Motion-blur |
| --- | --- | --- | --- | --- |
| SP | SP | | | |
| SP | SIFT | | | |
| SIFT | SP | | | |
| SIFT | SIFT | | | |

→ Detector level 손실 vs Descriptor level 손실 격리

### 표 D — Gating Ablation

| Gating Strategy | MAA @3px (avg) |
| --- | --- |
| Random | |
| Always SuperPoint+LG | |
| Decision Tree (ours) | |
| Oracle (upper bound) | |

---

## 5. Data Strategy — 2 Tier

### Tier 1 — Main Quantitative Evaluation: HPatches + Synthetic Augmentation

- HPatches 116 scenes × 5 pairs = 580 base pairs
- 합성 조건: Low-light, Motion-blur, Gaussian noise, Combined
- Severity 3 levels × 4 conditions → 원본 포함 **약 7,500 pairs**
- GT homography는 sub-pixel 정확도 보장 (HPatches 제공)

### Tier 2 — Qualitative Demo: 자체 촬영

- 10–15 페어 (간판, 책표지, 화이트보드 등 평면 위주)
- GT 없이 데모 영상용
- Hybrid의 condition 판단과 method 선택 과정 시각화

### 5.1 Synthetic Augmentation 상세

| Condition | 구현 |
| --- | --- |
| Low-light | Gamma correction (γ=2.0–3.5) + Poisson shot noise + Gaussian read noise |
| Motion-blur | 비선형 handshake trajectory PSF + convolution |
| Gaussian noise | σ ∈ {5, 15, 25} |
| Combined | 위 augmentation 순차 적용 |

> ImageNet-C protocol (`imagecorruptions` 라이브러리)을 표준으로 인용.

---

## 6. Proposed Method — Condition-Aware Hybrid

### 6.1 System Architecture

```text
Input Image Pair
      │
      ▼
┌─────────────────────┐
│  Image Quality      │  brightness, Laplacian variance,
│  Assessor (IQA)     │  SNR 등 hand-crafted 통계
└────────┬────────────┘
         │ condition label
         ▼
┌─────────────────────┐
│  Method Selector    │  Decision Tree
│  (Gating)           │  (조건 → matcher 매핑)
└────────┬────────────┘
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
LoFTR                  SP + LightGlue
(저조도 / 블러 심할 때)   (일반 / 경미한 조건)
    │                         │
    └────────┬────────────────┘
             ▼
       Matches + Homography
```

### 6.2 Image Quality Assessor

세 가지 hand-crafted 통계로 조건을 판별한다:

| Feature | 측정 대상 | 설명 |
| --- | --- | --- |
| Mean brightness | 저조도 여부 | 전체 픽셀 평균 (grayscale) |
| Laplacian variance | 블러 여부 | 낮을수록 blur 심함 |
| Local std (patch) | 노이즈 여부 | 균일 영역 내 픽셀 분산 |

### 6.3 Method Selector (Decision Tree Gating)

```text
brightness < τ_b  OR  laplacian_var < τ_l
        │
       Yes → LoFTR
        │
       No  → SuperPoint + LightGlue
```

τ_b, τ_l 임계값은 HPatches validation split에서 grid search로 결정.

### 6.4 Design Rationale

- **LoFTR**: 저조도·블러 상황에서 keypoint 자체가 부족할 때, detector-free 방식이 sparse pipeline 대비 유리
- **SP + LightGlue**: 조건이 충분히 좋을 때 속도와 정확도 모두 우수
- 복잡한 학습 없이 표 A–C 분석 결과에서 직접 임계값 도출 → 분석과 제안이 연결됨

---

## 7. Project Structure (예정)

```text
.
├── data/
│   ├── hpatches/          # HPatches 원본
│   └── augmented/         # 합성 조건 생성 결과
├── matchers/
│   ├── sift_orb.py
│   ├── superpoint.py
│   ├── lightglue.py
│   ├── loftr.py
│   └── hybrid.py          # Condition-aware hybrid (본 기여)
├── eval/
│   ├── detector_eval.py   # Level 1
│   ├── descriptor_eval.py # Level 2
│   └── pipeline_eval.py   # Level 3
├── augment.py             # Synthetic augmentation
├── iqa.py                 # Image Quality Assessor
└── run_experiments.py
```

---

## 8. References

- DeTone et al., *SuperPoint: Self-Supervised Interest Point Detection and Description*, CVPRW 2018
- Sun et al., *LoFTR: Detector-Free Local Feature Matching with Transformers*, CVPR 2021
- Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023
- Balntas et al., *HPatches: A benchmark and evaluation of handcrafted and learned local descriptors*, CVPR 2017
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions*, ICLR 2019
