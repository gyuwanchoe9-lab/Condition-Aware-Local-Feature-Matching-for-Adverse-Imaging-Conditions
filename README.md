# Condition-Aware Local Feature Matching for Adverse Imaging Conditions

> 2026 컴퓨터비전 텀프로젝트 — 서울과학기술대학교 전기정보공학과

---

## What is this?

 카메라로 촬영한 **저조도·모션블러·노이즈 환경**에서 두 이미지 간 특징점 매칭 성능을 체계적으로 분석하고, 입력 영상의 상태를 자동으로 판별해 최적 matcher를 선택하는 **Condition-Aware Hybrid Matcher**를 제안합니다.

### 왜 이게 필요한가?

SuperPoint(DeTone et al., 2018)는 학습 시 blur·저조도·노이즈 augmentation을 사용했다고 명시하지만, **평가는 clean 이미지에서만** 진행했습니다. 실제 악조건에서 어떤 matcher가 얼마나, 그리고 **어느 부분(detector vs descriptor)이 먼저 무너지는지**에 대한 분석이 없습니다.

본 프로젝트는 이 빈틈을 다음 세 질문으로 채웁니다:

1. 악조건에서 각 matcher의 성능 하락 폭은 얼마나 되는가?
2. Detector가 먼저 무너지는가, Descriptor가 먼저 무너지는가?
3. 조건을 자동 판별해 최적 matcher를 선택하면 성능이 개선되는가?

---

## Methods

| 카테고리 | 모델 | 역할 |
| --- | --- | --- |
| Classical | SIFT, ORB | Baseline |
| Learned full | SuperPoint + Mutual NN | SP 논문 원본 셋업 |
| Modern learned | SuperPoint + LightGlue | 현재 SOTA |
| Detector-free | LoFTR | 다른 패러다임 비교 |
| Cross-combination | SP det + SIFT desc, SIFT det + SP desc | Detector vs Descriptor 분해 분석 |
| **Ours** | **Condition-Aware Hybrid** | IQA → SP+LG or LoFTR 자동 선택 |

---

## Key Results

### Detector Repeatability (580 pairs, HPatches)

| Detector | Normal | Low-light sev3 | Motion-blur sev3 | Combined sev3 |
| --- | --- | --- | --- | --- |
| DoG (SIFT) | 0.433 | 0.348 | 0.330 | 0.272 |
| FAST (ORB) | 0.625 | 0.557 | 0.492 | 0.433 |
| SuperPoint | 0.511 | 0.426 | 0.401 | 0.349 |

> FAST가 repeatability는 가장 높지만 이것이 매칭 성능으로 이어지지 않음 — SuperPoint 논문과 동일한 관찰.

### Full Pipeline MAA@3px

| Method | Normal | Low-light | Motion-blur | Combined |
| --- | --- | --- | --- | --- |
| SIFT | 0.357 | 0.242 | 0.273 | 0.216 |
| ORB | 0.306 | 0.234 | 0.227 | 0.170 |
| SP + NN | 0.364 | 0.281 | 0.264 | 0.210 |
| SP + LightGlue | 0.448 | 0.379 | 0.326 | 0.283 |
| LoFTR | - | - | - | - |
| **Hybrid (Ours)** | - | - | - | - |

> 전체 580 pairs 결과로 업데이트 예정

---

## Proposed Method — Condition-Aware Hybrid

```text
Input Image Pair
      │
      ▼
┌─────────────────────┐
│  Image Quality      │  brightness + Laplacian variance
│  Assessor (IQA)     │
└────────┬────────────┘
         │ normal / degraded
         ▼
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
SP + LightGlue           LoFTR
(조건 양호)             (저조도 / 블러)
    │                       │
    └──────────┬────────────┘
               ▼
         Matches + Homography
```

- **IQA**: brightness(저조도) + Laplacian variance(블러) 두 통계로 조건 판별
- **Gating**: Decision Tree, 임계값은 HPatches validation split에서 grid search로 결정
- **근거**: 악조건에서 keypoint가 부족할 때 detector-free(LoFTR)가 유리, 조건이 충분하면 SP+LG가 속도·정확도 모두 우수

---

## Data

- **HPatches** 116 scenes × 5 pairs = **580 pairs** (GT homography 포함)
- 합성 augmentation: Low-light / Motion-blur / Gaussian noise / Combined, severity 1–3

```bash
# HPatches 다운로드 (1.3GB)
cd data
wget https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
unzip hpatches-sequences-release.zip -d hpatches_tmp
mv hpatches_tmp/hpatches-sequences-release/* hpatches/
```

---

## Setup

```bash
pip install kornia git+https://github.com/cvg/LightGlue.git
```

---

## Run

```bash
# 전체 실험 + figure 자동 생성
python run_experiments.py

# 빠른 테스트 (10쌍)
python run_experiments.py --max_pairs 10

# figure만 재생성
python plot_results.py

# 매칭 결과 시각화
python visualize.py --scene i_ajuntament
```

결과는 `results/pipeline_{N}pairs_{timestamp}.csv`, figure는 `figures/`에 저장됩니다.

---

## Project Structure

```text
.
├── augment.py          # Synthetic augmentation (low-light, blur, noise)
├── iqa.py              # Image Quality Assessor
├── run_experiments.py  # 실험 실행
├── plot_results.py     # CSV → 논문용 figure 생성
├── visualize.py        # 이미지 매칭 결과 시각화
├── matchers/
│   ├── classical.py    # SIFT, ORB, Cross-combination
│   ├── learned.py      # SuperPoint, SP+LightGlue, LoFTR
│   └── hybrid.py       # Condition-Aware Hybrid (본 기여)
├── eval/
│   ├── metrics.py      # Repeatability, MMA, MAA, Inlier Ratio
│   ├── hpatches.py     # HPatches 데이터 로더
│   └── run_eval.py     # Level 1 / Level 3 평가
├── data/hpatches/      # HPatches 데이터 (gitignore)
├── results/            # CSV 결과 (gitignore)
└── figures/            # 생성된 figure
```

---

## References

- DeTone et al., *SuperPoint*, CVPRW 2018
- Sun et al., *LoFTR*, CVPR 2021
- Lindenberger et al., *LightGlue*, ICCV 2023
- Balntas et al., *HPatches*, CVPR 2017
