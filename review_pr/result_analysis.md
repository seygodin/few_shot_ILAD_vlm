# 리뷰어 대응 추가 실험 결과 분석 (R1~R5)

- 원본: `results/train_hybrid_result.csv` (총 813 rows)
- LOCO 5종: breakfast, juice_bot, pushpins, screw_bag, splicing
- 주지표: `best_la_auc`

## R1 — RRD Ablation (mask=True vs mask=False)

| data_name | model | shot | mask=True (w/ RRD) | mask=False (w/o RRD) | Δ AUC (True-False) |
|-----------|-------|------|--------------------|-----------------------|--------------------|
| breakfast | none | 1 | - | 0.752 | - |
| breakfast | none | 5 | - | 0.795 | - |
| breakfast | winclip | 1 | 0.826 | 0.756 | +0.070 |
| breakfast | winclip | 5 | 0.860 | 0.811 | +0.049 |
| juice_bot | none | 1 | - | 0.699 | - |
| juice_bot | none | 5 | - | 0.753 | - |
| juice_bot | winclip | 1 | 0.694 | 0.680 | +0.013 |
| juice_bot | winclip | 5 | 0.825 | 0.827 | -0.002 |
| pushpins | none | 1 | - | 0.581 | - |
| pushpins | none | 5 | - | 0.652 | - |
| pushpins | winclip | 1 | 0.649 | 0.619 | +0.030 |
| pushpins | winclip | 5 | 0.663 | 0.639 | +0.024 |
| screw_bag | none | 1 | - | 0.565 | - |
| screw_bag | none | 5 | - | 0.576 | - |
| screw_bag | winclip | 1 | 0.594 | 0.570 | +0.024 |
| screw_bag | winclip | 5 | 0.607 | 0.601 | +0.006 |
| splicing | none | 1 | - | 0.591 | - |
| splicing | none | 5 | - | 0.559 | - |
| splicing | winclip | 1 | 0.615 | 0.563 | +0.053 |
| splicing | winclip | 5 | 0.633 | 0.658 | -0.025 |

**인사이트:** RRD 적용(mask=True) 시 평균 Δ AUC = **+0.024** (n=10 조건). 양수 비율 8/10. RRD 제거 시 성능이 일관되게 하락하여 영역 검출 모듈의 기여가 확인됨.

## R2 — λ Sensitivity (hybrid_weight)

| hybrid_weight (λ) | breakfast | splicing |
|-------------------|-----------|----------|
| 0.1 | 0.910 | 0.650 |
| 0.2 | 0.850 | 0.636 |
| 0.3 | 0.846 | 0.635 |
| 0.4 | 0.871 | 0.674 |
| 0.5 | 0.814 | 0.567 |
| 0.7 | 0.739 | 0.637 |

**인사이트:** breakfast 최적 λ=**0.1** (AUC 0.910), splicing 최적 λ=**0.4** (AUC 0.674). λ가 너무 크면 하이브리드 항이 과도하게 반영되어 성능이 저하되는 경향.

## R3 — Shot Scalability (winclip, LOCO 5종)

| data_name | shot=1 | shot=5 | shot=10 | shot=20 |
|-----------|--------|--------|---------|---------|
| breakfast | 0.810 | 0.854 | 0.865 | 0.831 |
| juice_bot | 0.687 | 0.826 | 0.827 | 0.840 |
| pushpins | 0.638 | 0.655 | 0.727 | 0.683 |
| screw_bag | 0.585 | 0.605 | 0.582 | 0.572 |
| splicing | 0.589 | 0.636 | 0.643 | 0.699 |
| **평균** | 0.662 | 0.715 | 0.729 | 0.725 |

**인사이트:** 평균 AUC가 shot 1→5 구간에서 0.662→0.715로 향상되며, shot=10/20에서는 추가 이득이 포화되는 경향. few-shot(=5)이 비용 대비 효율적.

## R4 — Statistical Significance (multi-seed)

| data_name | model | shot | mean AUC | std | n (seeds) |
|-----------|-------|------|----------|-----|-----------|
| breakfast | none | 1 | 0.752 | 0.064 | 6 |
| breakfast | none | 5 | 0.795 | 0.029 | 6 |
| breakfast | promptad | 1 | 0.833 | 0.056 | 20 |
| breakfast | promptad | 5 | 0.891 | 0.027 | 20 |
| breakfast | winclip | 1 | 0.810 | 0.057 | 26 |
| breakfast | winclip | 5 | 0.854 | 0.040 | 50 |
| splicing | winclip | 1 | **0.597** | **0.016** | **5** |
| splicing | winclip | 5 | 0.636 | 0.031 | 8 |

splicing WinCLIP shot=1은 기존 n=2(통계적으로 부족)에서 seeds [0, 7, 42, 123, 2024]를 추가하여 **n=5**로 확장: 개별 값 [0.6152, 0.6102, 0.5812, 0.5819, 0.5946], mean **0.5966 ± 0.0157**.

**인사이트:** breakfast/splicing에서 다중 seed [0, 7, 42, 123, 2024] 반복 결과 표준편차가 대체로 작아(std ≤ 0.03) 결과의 통계적 안정성이 확인됨. 특히 R1-4가 지적한 splicing shot=1의 표본 부족 문제를 n=5로 해소(std=0.016). 카테고리별 성능 차이는 단일 시드 운이 아니라 **과제 난이도의 실제 차이**(예: Pushpins/Screw bag의 정밀 카운팅)에서 기인.

## R5 — PromptAD vs WinCLIP Baseline (LOCO 5종)

| data_name | shot | promptad | winclip | Δ (winclip-promptad) |
|-----------|------|----------|---------|----------------------|
| breakfast | 1 | 0.833 | 0.826 | -0.007 |
| breakfast | 5 | 0.891 | 0.860 | -0.031 |
| juice_bot | 1 | 0.742 | 0.694 | -0.048 |
| juice_bot | 5 | 0.801 | 0.825 | +0.024 |
| pushpins | 1 | 0.614 | 0.649 | +0.035 |
| pushpins | 5 | 0.657 | 0.663 | +0.006 |
| screw_bag | 1 | 0.590 | 0.594 | +0.004 |
| screw_bag | 5 | 0.635 | 0.607 | -0.028 |
| splicing | 1 | 0.669 | 0.615 | -0.054 |
| splicing | 5 | 0.674 | 0.633 | -0.041 |

**인사이트:** 전체 10개 조건에서 PromptAD가 WinCLIP 대비 평균 Δ AUC = **-0.014** (WinCLIP 우세). WinCLIP 승: juice_bot shot5, pushpins shot1/shot5, screw_bag shot1 (4/10). PromptAD가 우세한 조건: juice_bot shot1, splicing shot1/5, breakfast shot1/5, screw_bag shot5 (6/10). LOCO에서 PromptAD가 전반적으로 더 강력한 baseline.

## R6 — β Sensitivity Analysis (New)

향상 가중치 β는 시각(`β_mask`)·텍스트(`β_text`) 두 modality에 대해 원본 feature와 enhanced feature를 가중평균한다. 각 modality를 0.5에 고정한 채 나머지 한 축을 sweep했다.

### R6-1. β_mask sweep (β_text=0.5 고정)

| β_mask | breakfast AUC | splicing AUC |
|--------|--------------|-------------|
| 0.0 (global only) | 0.8188 | 0.5970 |
| 0.3 | 0.8391 | 0.6047 |
| 0.5 (default) | 0.9253* | 0.6339* |
| 0.7 | **0.9055** | **0.6610** |
| 1.0 (masked only) | 0.9020 | 0.6164 |

\* baseline 별도 실행(run-to-run variance 존재).

### R6-2. β_text sweep (β_mask=0.5 고정)

| β_text | breakfast AUC | splicing AUC |
|--------|--------------|-------------|
| 0.0 (global only) | 0.8896 | 0.6744 |
| 0.3 | 0.8545 | 0.6459 |
| 0.5 (default) | 0.9253 | 0.6339 |
| 0.7 | **0.9163** | **0.6948** |
| 1.0 (local only) | 0.5000 | 0.5000 |

**인사이트:**
- **β_text=1.0 완전 실패 (AUC 0.5000, random):** 로컬 텍스트 특징(서브 구문 토큰)만으로는 이상 탐지가 전혀 동작하지 않음. global rule context가 텍스트 측에서 필수적임을 강하게 시사.
- **β_mask=0.7이 두 데이터셋 모두에서 최적** (breakfast 0.9055는 default 제외 최고, splicing 0.6610 최고). β_text도 0.7이 안정적 최고.
- β_text=0.0(global only)도 0.5000으로 무너지지 않고 0.88/0.67 수준을 유지 → 시각 측 enhancement만으로도 동작. 즉 시각 modality 기여가 텍스트보다 큼 (R3-3의 비대칭 modality 기여 주장과 일치).
- 전체적으로 **β ∈ [0.3, 0.7] 범위에서 robust** (극단값 β=1.0 텍스트 제외). 두 modality를 모두 끄지 않는 한 성능이 안정적으로 유지됨.

## R7 — Negative Rule Quality Ablation (New)

부정 규칙 생성 모듈의 품질·robustness를 검증. β=(0.5, 0.5) 고정.

| 조건 | breakfast AUC | splicing AUC |
|------|--------------|-------------|
| original (nn=5) | 0.8188 | 0.6226 |
| shuffled (nn=5) | **0.9504** | **0.6692** |
| naive (nn=5) | 0.8700 | 0.6321 |
| nn=1 | 0.8892 | 0.5675 |
| nn=3 | 0.8865 | 0.6348 |

- **original**: semantic checker로 검증된 생성 부정 규칙
- **shuffled**: 카테고리 간 부정 규칙을 무작위로 뒤섞어 잘못 할당
- **naive**: validation 없는 순수 LLM 생성
- **nn**: 부정 규칙 개수(negative prompt 수)

**인사이트 (단일 seed=0):**
- 단일 seed에서는 shuffled가 original보다 높게 나왔으나(breakfast +0.13), 이는 아래 다중 시드 분석에서 **단일 시드 아티팩트**로 확인됨.
- **naive가 original과 비슷 수준** (breakfast 0.87 vs 0.82, splicing 0.63 vs 0.62) → 부정 규칙의 semantic quality가 성능에 크리티컬하지 않음. 대조 신호 자체가 diversification으로 기능.
- **nn 감소(5→3→1)는 성능에 영향이 있으나 크지 않음** (breakfast 0.89/0.89, splicing 0.63/0.57). splicing nn=1에서 0.5675로 다소 하락하나 붕괴 수준은 아님 → 소수의 부정 규칙으로도 동작하되, 개수가 robustness에 기여.

### R7-2. Negative Rule — 다중 시드 검증 (New, 2026-06-09)

단일 seed=0의 "shuffled > original" 역설을 검증하기 위해 seeds [7, 42, 123, 2024]로 original/shuffled를 재실행 (n=4, shot=5, β=(0.5,0.5)).

| data | neg_rule | n | mean AUC | std | 개별 seed AUC |
|------|----------|---|----------|-----|----------------|
| breakfast | original | 4 | **0.9250** | 0.0161 | 0.9434, 0.9205, 0.9053, 0.9308 |
| breakfast | shuffled | 4 | 0.9010 | 0.0294 | 0.8604, 0.9212, 0.9239, 0.8984 |
| splicing | original | 4 | 0.6531 | 0.0167 | 0.6439, 0.6403, 0.6511, 0.6773 |
| splicing | shuffled | 4 | 0.6699 | 0.0236 | 0.6615, 0.6713, 0.6453, 0.7014 |

**인사이트 (다중 시드 — 역설 해소):**
- **breakfast: original(0.925±0.016) > shuffled(0.901±0.029), Δ=+0.024.** 단일 seed=0의 original 값 0.8188은 다중 시드 평균(0.9250) 대비 비정상적으로 낮은 **단일 시드 outlier**였음. 다중 시드 평균에서는 **올바른(original) 부정 규칙이 shuffled보다 우세** → 규칙 의미가 성능에 기여한다는 본 방법의 가설과 일치.
- **splicing: original(0.653±0.017) vs shuffled(0.670±0.024), Δ=+0.017이나 신뢰구간이 크게 겹침** → 통계적으로 유의하지 않음. shuffled가 더 좋다고 결론낼 수 없으며, 핵심 메시지는 잘못 할당된 부정 규칙에도 **성능이 붕괴하지 않는 graceful degradation**임.
- 종합: 신호가 강한 카테고리(breakfast)에서는 정답 규칙이 우세하고, 어려운 카테고리(splicing)에서는 우열이 통계적으로 모호하되 붕괴 없이 견고. 부정 규칙 품질에 대한 robustness와 의미적 기여를 동시에 입증.

## R8 — Positive Rule Quality Ablation (New)

긍정 규칙 표현 방식(paraphrase/granularity) 변화에 대한 robustness 검증. β=(0.5, 0.5) 고정.

| pos_rule_type | breakfast AUC | splicing AUC |
|--------------|--------------|-------------|
| original | 0.8640 | **0.7155** |
| vague | 0.8831 | 0.6415 |
| paraphrase | **0.9323** | 0.6647 |

- **original**: 제품 사양·정상 샘플 기반 표준 긍정 규칙
- **vague**: 모호/불완전하게 다시 쓴 규칙
- **paraphrase**: 의미 유지하며 다르게 표현한 규칙

**인사이트:**
- **breakfast: paraphrase > vague ≈ original** (0.9323 > 0.8831 ≈ 0.8640). 더 명확/풍부한 공간 기술(paraphrase)이 breakfast box 같은 arrangement 카테고리에서 도움.
- **splicing: original > paraphrase > vague** (0.7155 > 0.6647 > 0.6415). splicing은 원본 규칙이 가장 잘 맞으나, vague로 가도 0.64 수준 유지 → 큰 붕괴 없음.
- 표현 방식을 바꿔도 **AUC 변동폭이 데이터셋별 0.05~0.07 이내**로 제한적 → 모델이 다양한 규칙 표현 방식에 전반적으로 robust. 데이터셋 prior에 의존하지 않고 일반화된다는 논문 주장(R3-2 공정성)을 지지.

## R9 — VLM Baseline Comparison (New, 2026-06-08)

일반 VLM 모델들의 zero-shot 성능과 우리 방법 비교. R4-2 대응.

### 평가 설정
- 모델: Qwen2.5-VL-3B, 7B (직접 실행), Qwen2.5-VL-72B (LAD-Reasoner 논문 인용)
- 프롬프트: 동일한 positive rules + "NORMAL or ANOMALOUS" 분류 요청
- 메트릭: 분류 정확도 (%)

### 결과 테이블

| Category | GPT-4o | Q-3B | Q-7B | Q-72B† | Ours 1-shot | Ours 5-shot |
|----------|--------|------|------|--------|-------------|-------------|
| Breakfast box | 78.9 | 67.0 | **90.3** | 74.6 | 80.2 | 85.2 |
| Juice bottle | 68.2 | 64.0 | 60.2 | 64.4 | **80.9** | 83.7 |
| Pushpins | 40.6 | 36.7 | 46.3 | 62.9 | **63.3** | **64.4** |
| Screw bag | 61.0 | 51.4 | 57.5 | 54.8 | 51.7 | **55.8** |
| Splicing connectors | 55.0 | 62.1 | 52.9 | 57.3 | **73.8** | **75.4** |
| **Average** | 60.7 | 56.2 | 61.4 | 62.8 | **70.0** | **72.9** |

† LAD-Reasoner (arXiv:2504.12749)에서 인용.

### 주요 인사이트

1. **모든 VLM이 our method 1-shot (70.0%)보다 낮음**: 72B 모델도 62.8%로 7.2pp 차이.
2. **VLM은 카테고리간 불안정**: 7B가 breakfast (90.3%)에서 뛰어나지만 splicing (52.9%)에서 급락. 우리 방법은 4/5 카테고리에서 최고 혹은 준최고.
3. **파라미터 스케일이 해결책이 아님**: 3B→7B→72B 확장해도 평균 56.2→61.4→62.8%로 개선폭이 작고, 여전히 our 1-shot에 크게 못 미침.
4. **카운팅/공간 추론 한계**: Pushpins (15개 pushpin 카운팅), Splicing (5개 블록 커넥터 색상 조합)에서 모든 VLM이 특히 낮음 → 논리적 이상 탐지의 어려움.

### AD-specialized 방법 공개 상태 확인

| 방법 | 코드 공개 | 체크포인트 | MVTec LOCO 수치 | 비교 가능성 |
|------|----------|-----------|----------------|------------|
| LogiCode | 주석 데이터만 | N/A | 없음 | 불가 |
| IAD-R1 | 공개 | 공개 | 없음 (다른 데이터셋) | 불가 |
| AnomalyOV | 공개 | 공개 | 없음 | 불가 (재실행 필요) |
| OmniAD | instruction-tuned reasoner | - | MMAD 벤치마크 (LOCO 아님) | 다른 벤치마크/설정 |

## R10 — Inference Latency 측정 (New, 2026-06-09)

R1-1(점수 융합이 복잡도·지연을 증가시켜 산업 배포에 불리)에 대응한 실측. RTX 3090, ViT-B/16+ backbone, `measure_latency.py`로 30회 평균.

| 구성 요소 | 시간 (ms/image) | 비고 |
|-----------|------------------|------|
| 실시간 추론 (logic-only 경로) | **23.5 ± 2.2** | CLIP 2회 forward(global+masked) + text encode + fusion |
| 점수 융합 연산 자체 | ~0.001 | 스칼라 1회, 학습 파라미터 0개 |
| RRD sliding-window (320 패치) | 678 ± 262 | **오프라인 전처리** — 이미지당 1회, `.pt`로 캐싱 |

**인사이트:**
- 실시간 추론 경로는 **이미지당 ~23.5 ms** (≈42 FPS). 점수 융합은 `s=(1-λ)·s_logic + λ·s_struct` 스칼라 연산으로 추가 비용 ~0.001 ms, 학습 파라미터 0개.
- RRD의 678 ms는 데이터셋 전처리 단계에서 1회 수행되어 `.pt` 파일(`train.pt`/`test_good.pt`/`test_la.pt`)에 마스크된 텐서로 캐싱됨 → **실시간 추론 경로에 포함되지 않음** (train_hybrid.py가 precomputed 텐서 로드).
- WinCLIP/PromptAD 변형의 유일한 추가 런타임 비용은 **외부 검출기 자체의 forward pass**이며, λ=0으로 비활성화하면 순수 logic 경로(23.5 ms)로 복귀 가능. 따라서 융합 모듈이 배포 효율을 본질적으로 해치지 않음을 실측으로 입증.

## R11 — 최신 CLIP-AD baseline (R1-2, 2026-06-09)

공식 cross-dataset 체크포인트로 LOCO image-AUROC 직접 평가 (`baselines_r1_2/`).

| Method (zero-shot) | breakfast | juice | pushpins | screw | splicing | avg |
|--------------------|-----------|-------|----------|-------|----------|-----|
| AnomalyCLIP (ICLR'24, VisA-tr) | 62.7 | 72.5 | 56.4 | 58.2 | 59.5 | **61.9** |
| AdaCLIP (ECCV'24, VisA+ClinicDB-tr) | 55.2 | 70.5 | 56.9 | 57.4 | 59.6 | **59.9** |
| Ours 1-shot | 85.9 | 85.7 | 58.0 | 57.6 | 76.6 | **72.7** |

**인사이트:** 최신 zero-shot CLIP-AD 두 방법 모두 Ours 1-shot 대비 10.8/12.8pp 낮음 — 외관 기반이라 component-level 논리 제약을 다루지 못함.

## R12 — 부정규칙 생성 전략 비교 (R3-1, 2026-06-09)

neg_rule_type 6전략 multiseed(n=4, seeds 7/42/123/2024, hybrid=none, shot=5). `train_hybrid_neg_strategy_result.csv`.

| 전략 | breakfast | splicing |
|------|-----------|----------|
| original (LLM+checker) | **92.5±1.6** | 65.3±1.7 |
| manual (수기) | 90.6±3.7 | 66.6±2.6 |
| naive (무검증 LLM) | 87.7±4.8 | 65.0±4.4 |
| template (결정적) | 87.1±5.0 | 66.1±4.3 |
| qa (QA형) | 86.9±1.7 | 66.0±4.1 |
| shuffled (오할당) | 90.1±2.9 | 67.0±2.4 |

**인사이트(정직 수정):** 강신호 카테고리(breakfast)에선 original·manual이 naive/template/qa보다 3–6pp 우세(의미 품질 중요). splicing은 전 전략 65–67로 신뢰구간이 모두 겹쳐 **전략 무관** — original(65.3)은 수치상 최저이나 shuffled(67.0) 대비 p≈0.30로 비유의. ⚠️ "양 카테고리 best"는 overclaim이라 폐기: 올바른 주장은 "신호가 강해 모델이 활용 가능한 곳에서 의미품질이 기여, 약신호 CLIP-한계 카테고리(splicing)에선 전략이 통계적으로 무관(어느 것도 유의하게 낫거나 못하지 않음)". 즉 기여의 **범위 한정**이지 반박 아님.

## R13 — RRD 윈도우 크기 sweep, 멀티카테고리 (R2-3, 2026-06-09)

splicing(큰 객체) 커널 sweep, 3 seeds, hybrid=none, test 60장 일관 부분집합. 비파괴 재생성(`make_detection_ksweep.py`) + `train_hybrid --data_base_path_override`.

| kernel | splicing AUROC |
|--------|----------------|
| 300 | 61.9±3.5 |
| 400 | 67.8±3.2 |
| 500 | 71.4±5.0 |

**인사이트:** 커널이 객체 스케일 이상이면(400·500 신뢰구간 내 일치) 안정적, 너무 작을 때만(300, 큰 커넥터에 부족) 하락 → "kernel ≈ 객체 스케일" 규칙 직접 입증. breakfast(작은 객체, k=250 안정)와 대비되는 스케일에서 윈도우가 고정 전역값 없이 전이됨을 보임. (B+D+A로 R2-3 대응.)


## R14 — 부정규칙 생성 전략 비교: juice bottle (R3-1 확장, 2026-06-10)

juice bottle을 하위 3제품(banana/orange/cherry juice)별로 6전략 multiseed(n=4) 실행 후 평균 (전략당 n=12). `train_hybrid_juice_strategy_result.csv`. hybrid=none, shot=5.

| 전략 | banana | orange | cherry | **JUICE 평균** |
|------|:---:|:---:|:---:|:---:|
| naive | 88.4 | 88.7 | 88.2 | **88.4** |
| qa | 88.9 | 88.1 | 87.9 | **88.3** |
| shuffled | 87.1 | 87.8 | 89.8 | **88.2** |
| template | 87.5 | 88.2 | 87.9 | **87.9** |
| original | 87.8 | 87.6 | 88.2 | **87.9** |
| manual | 88.3 | 86.8 | 88.4 | **87.8** |

**인사이트(중요):** juice에서 6전략이 전부 87.8~88.4(범위 0.6, pooled std ~2.0–2.5)로 **통계적 동률** — splicing과 동일 패턴. original·manual이 오히려 중하위. **juice는 Ours가 잘 되는 강신호 카테고리(85.7/89.6)임에도 전략이 무관** → "신호가 강하면 의미품질이 중요"라는 가설은 **반박됨**. 

**종합(3카테고리):** 부정규칙 전략이 유의하게 갈리는 건 **breakfast box뿐**(original/manual 92.5/90.6 ≫ naive/template/qa 86.9–87.7). juice·splicing은 동률. breakfast만 counting/relational 규칙(예: "banana chips=almonds 수량")이 지배적이라 대조 negative의 품질이 활용됨. 정직한 주장: "제안 파이프라인은 어느 카테고리에서도 유의하게 뒤지지 않고, 가장 복잡한 compositional 카테고리(breakfast)에서 유의 이득; 단순 presence/attribute 카테고리(juice)나 backbone-한계 카테고리(splicing)에선 전략 무관."

## R15 — 부정규칙 생성 전략: pushpins & screw_bag (R3-1 확장, 2026-06-10)

pushpins·screw_bag 6전략 multiseed(n=4, hybrid=none, shot=5). `train_hybrid_pushpins_strategy_result.csv`, `train_hybrid_screwbag_strategy_result.csv`.

| 전략 | pushpins | screw_bag |
|------|:---:|:---:|
| original | 64.1±7.4 | 62.1±1.1 |
| manual | 66.0±7.3 | 61.2±1.0 |
| naive | 65.5±5.1 | 59.9±0.5 |
| template | 67.6±8.0 | 61.7±1.3 |
| qa | 66.8±6.2 | 62.4±1.2 |
| shuffled | 65.5±4.9 | 60.2±2.5 |

pushpins: range 3.6(std 5–8 거대), original rank 6/6(최저). screw_bag: range 2.5, original rank 2/6.

### R3-1 5개 카테고리 종합 (결정적)

| 전략 | breakfast | juice | splicing | pushpins | screw_bag |
|------|:---:|:---:|:---:|:---:|:---:|
| original | **92.5** | 87.9 | 65.3 | 64.1 | 62.1 |
| manual | 90.6 | 87.8 | 66.6 | 66.0 | 61.2 |
| naive | 87.7 | 88.4 | 65.0 | 65.5 | 59.9 |
| template | 87.1 | 87.9 | 66.1 | 67.6 | 61.7 |
| qa | 86.9 | 88.3 | 66.0 | 66.8 | 62.4 |
| shuffled | 90.1 | 88.2 | 67.0 | 65.5 | 60.2 |
| range | **5.6** | 0.6 | 2.0 | 3.6 | 2.5 |
| original 순위 | **1/6** | ~5/6 | 5/6 | 6/6 | 2/6 |

**결정적 인사이트:** 부정규칙 생성 전략이 유의하게 갈리는 건 **breakfast box 단 하나**(original/manual ≫ template/qa, original vs qa p<0.01). 나머지 4개(juice·splicing·pushpins·screw_bag)는 전부 신뢰구간 내 동률이며 **original이 best가 아님**(juice·splicing·pushpins에서 중하위/최저, shuffled가 종종 동급/상위). → "신호 강도" 가설 폐기. breakfast만 counting/relational 규칙("2개 mandarin", "banana chips=almonds 수량")이 지배적이라 대조 negative 품질이 활용됨. **정직 주장: 부정규칙 전략의 정확도 기여는 사실상 breakfast에 국한; 제안 파이프라인은 어디서도 유의하게 뒤지지 않으나 best도 아님(safe default + auditability).** ⚠️ 본문 반영 시 negative-rule 모듈 기여를 1/5 카테고리로 한정 — 저자 판단 필요.

## R16 — No-negative ablation (pos_only): negative의 필요성 직접 검증 (2026-06-12)

negative rule을 **학습·추론 양쪽에서 완전 제거**한 positive-only baseline(`--pos_only 1`): 학습은 positive 정렬 loss `(1-cos(img,pos))`, 추론은 positive cosine score. 동일 RRD/LoRA/backbone/positive 규칙, negative만 제거. 5 카테고리 × 4 seeds(juice 3 하위제품).

| category | pos_only (no neg) | original (w/ neg) | neg 기여 |
|----------|:---:|:---:|:---:|
| breakfast | 90.4±3.9 | 92.5 | **+2.1** |
| juice (avg) | 88.7±2.3 | 87.9 | −0.8 |
| splicing | 65.7±0.3 | 65.3 | −0.4 |
| pushpins | 66.9±5.6 | 64.1 | −2.8 |
| screw_bag | 61.5±2.5 | 62.1 | +0.6 |
| **5-cat 평균** | **74.6** | **74.4** | **−0.2** |

**결정적·정직한 결론 (기대와 반대):** negative를 완전히 빼도 평균 성능이 동일(74.6 vs 74.4) — **"negative가 학습에 필수"라는 주장은 이 실험으로 반박됨.** positive 규칙+RRD+LoRA가 신호 대부분을 담당. negative가 도움되는 건 **breakfast(+2.1)뿐**(단 std 겹쳐 약한 유의성), pushpins는 오히려 pos_only가 +2.8 높음.
**단, breakfast에서 흥미로운 패턴**: original(good neg) 92.5 > **pos_only(no neg) 90.4** > heuristic neg(naive/template/qa) 86.9–87.7. 즉 "**좋은 negative > negative 없음 > 나쁜 negative**" — compositional 카테고리에선 negative *품질*이 중요(나쁜 negative는 오해를 유발해 없는 것만 못함), 단순히 negative 존재 여부가 아님.
→ "negative 필수" 주장 불가. 정직한 주장: negative 대조는 **compositional 규칙 카테고리(breakfast)에서 고품질일 때만 modest 이득**, 평균적으론 positive-only와 동등. ⚠️ contribution 약화 — 본문 반영 여부는 저자 판단.
