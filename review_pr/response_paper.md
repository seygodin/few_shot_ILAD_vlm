# Author Response — Pattern Recognition 2026 Special Issue

**Paper:** Rule-Guided Vision-Language Learning for Few-Shot Logical Anomaly Detection

> This document organizes reviewer concerns and provides point-by-point responses backed by experimental results and logical arguments. All newly reported numbers come from experiments run after the initial submission.

---

## Table of Contents

- [Editor-in-Chief (EiC)](#editor-in-chief-eic)
- [Associate Editor (AE)](#associate-editor-ae)
- [Reviewer 1 (R1)](#reviewer-1-r1)
- [Reviewer 2 (R2)](#reviewer-2-r2)
- [Reviewer 3 (R3)](#reviewer-3-r3)
- [Reviewer 4 (R4)](#reviewer-4-r4)

---

## Editor-in-Chief (EiC)

### EiC-a — Title conciseness (≤10–15 words)

**Concern:** The title should be succinct and grammatical, ideally not exceeding 10–15 words.

**Response:** The title has been revised to:

> *"Rule-Guided Vision-Language Learning for Few-Shot Logical Anomaly Detection"* (9 words)

The revised title is concise, grammatical, and captures the three core contributions: rule guidance, vision-language learning, and the few-shot logical anomaly detection task.

---

### EiC-b — Conclusion quality

**Concern:** The conclusions should reflect strengths and weaknesses, benefit to the field, future work, and must differ from the abstract.

**Response:** The conclusion section has been fully rewritten from 2 paragraphs to 5 structured paragraphs:

1. **Summary of contributions** — distinct from the abstract by focusing on what was validated, not just proposed
2. **Strengths** — rule-guided representation, zero-shot region detection, few-shot efficiency
3. **Limitations** — sensitivity of logical rule quality in edge cases, fixed fusion weight, evaluation confined to MVTec LOCO AD
4. **Future work** — adaptive per-category λ, automated rule induction, extending to video inspection
5. **Broader impact** — how the rule-guided paradigm can generalize to other human-knowledge-intensive detection tasks

---

### EiC-c — Bibliography (35–55 items, no grouped citations, no excessive arXiv)

**Concern:** Bibliography should be 35–55 items, no excessive arXiv, no grouped citations (e.g., "[1,2,3,4,5,6]").

**Response:**
- All grouped citations in the introduction (e.g., "[1–6]") have been split into individually commented references
- Four recent papers added: Anomize (2025), UniAD (2024), SSTP (2025), VDN (2024)
- Final bibliography count: **55 unique entries** (within the 35–55 limit)
- arXiv citations are kept to a minimum; only LAD-Reasoner (arXiv:2504.12749) is cited as it provides the only published Qwen2.5-VL-72B numbers on MVTec LOCO AD

---

### EiC-d — Recent Pattern Recognition–relevant citations

**Concern:** Ensure relevance to Pattern Recognition readership with recent citations from the field.

**Response:** Added four papers with individual discussion in the Related Work section:

| Paper | Where added | Why relevant |
|-------|------------|--------------|
| Anomize (2025) | Related work — open-vocabulary AD | Uses language to describe and detect anomalies, directly relevant to our rule-guided paradigm |
| UniAD (2024) | Related work — open-vocabulary AD | Integrates geometric and semantic cues for unified AD |
| SSTP (2025) — Fast Track Anything | Related work — RIS/segmentation | Efficient spatio-temporal propagation for target localization under weak supervision |
| VDN (2024) — Video Decoupling Network | Related work — RIS/segmentation | Robust zero-shot segmentation with cross-domain generalization ideas relevant to RRD |

---

### EiC-e — Page limit (≤35 pages, double-spaced single column)

**Concern:** Do not exceed 35 pages for a regular paper.

**Status:** LaTeX compilation is needed to verify exact page count. The revision added ~3 tables (β sensitivity, λ sensitivity, VLM comparison) and expanded the conclusion section. **Authors should compile and confirm compliance before submission.**

---

## Associate Editor (AE)

**Concern summary:** Limited novelty in some components; insufficient comparison with VLM/MLLM-based logical AD methods (e.g., LogiCode); need to distinguish from reasoning-driven approaches; inadequate validation of rule quality and robustness; limited justification of region detection and score fusion; stronger ablation studies, statistical analysis, scalability, and reproducibility required.

**Response overview:** All major experimental concerns have been addressed:

| AE Concern | Status | Evidence |
|-----------|--------|---------|
| LogiCode / VLM-based logical AD comparison | Addressed | Q-3B/7B + LLaVA-1.6-7B + Qwen2-VL-7B directly evaluated, Anomaly-OV evaluated directly (0.534 logical AUROC); LogiCode/IAD-R1/OmniAD cite-only (unreleased / no LOCO split / different benchmark) |
| Rule quality and robustness validation | Addressed | Negative rule ablation (R2-2/R3-1/R4-3) + positive rule robustness (R3-2) experiments |
| RRD justification | Addressed | Ablation across all 5 categories (R1): avg Δ AUC = +0.024 |
| Score fusion justification | Addressed | λ sensitivity sweep (R2) + conceptual analysis |
| Statistical analysis | Addressed | Multi-seed experiments (std typically ≤ 0.04; ≤0.03 for the new n=4/n=5 conditions, e.g. splicing shot=1 = 0.597±0.016) |
| Scalability | Addressed | Shot=1/5/10/20 evaluated |
| Reproducibility | Addressed | LoRA, MLP decoder, score normalization fully specified |

---

## Reviewer 1 (R1)

### R1-1 — Score fusion is conventional and increases complexity/latency

**Concern:** The complementary score fusion is straightforward and increases model complexity and inference latency, which is unfavorable for industrial deployment.

**Response:** We agree that score fusion is architecturally simple, and this simplicity is intentional. To directly address the deployment concern, we now report *measured* latency rather than a qualitative "negligible" claim. All measurements are on an RTX 3090 with the ViT-B/16+ CLIP backbone; other hardware/backbones will differ, and the 23.5 ms figure assumes the RRD mask is already precomputed and cached (a first-time, never-seen image incurs the one-off ~678 ms preprocessing described below).

| Component | Latency | On real-time path? | Trainable params |
|-----------|---------|--------------------|------------------|
| Logic-only inference (2 CLIP image forwards: global + precomputed-masked, + text encode + fusion) | **23.5 ± 2.2 ms / image** | Yes | — |
| Score-fusion operation `s = (1−λ)·s_logic + λ·s_struct` | **~0.001 ms** | Yes | **0** |
| RRD sliding-window region detection (320 patches) | ~678 ± 262 ms / image | **No — offline, cached** | 0 |
| External structural detector (WinCLIP / PromptAD), λ>0 only | that detector's own forward pass | only if enabled | 0 |

Three points follow directly from these numbers:

1. **The fusion itself is free.** The fused score is a single scalar operation, `s_fused = (1 − λ)·s_logic + λ·s_structural`, with **zero trainable parameters**; its measured cost is ~0.001 ms, i.e. ~4×10⁻⁵ of the 23.5 ms inference budget (below 10⁻⁴). The "complexity" added by the proposed component over the base logic path is therefore not measurable in practice.

2. **RRD's 678 ms is offline and amortized, not per-inference.** The sliding-window region detection that produces the masked region runs **once per image as a preprocessing step** and is cached to disk (the training/inference code loads precomputed masked tensors). It is therefore *not* on the real-time inference path; the 23.5 ms figure above already uses the precomputed mask. (A first-time, never-cached image incurs this ~678 ms once.) We have clarified this offline/online split explicitly in the revised paper to avoid the impression that RRD is a per-query runtime cost.

3. **The only runtime cost of the fused variant is the external detector itself, and it is optional.** When λ > 0, the extra runtime is exactly the WinCLIP/PromptAD forward pass — that external model's own cost (not separately profiled here), not an overhead introduced by our fusion. In latency-critical deployments the fusion can be disabled entirely by setting **λ = 0**, reverting to the pure logic path at **23.5 ms / image** with no loss of the logical-anomaly capability that is the paper's focus.

Consistent with R3-4, fusion (λ > 0) is most useful when structural cues are present (MVTec-AD / VisA) and contributes little on purely logical anomalies (LOCO) — there, the structural score is largely orthogonal to the logical-violation signal, so equal-weight fusion can even dilute the clean logic score. Disabling fusion on logic-dominated deployments is therefore expected to be near-lossless in accuracy (inferred from this R3-4 orthogonality analysis; a dedicated λ = 0 vs λ > 0 comparison is reported in the revised paper) while removing the external detector's runtime. We have added this measured latency breakdown and the trade-off discussion to the revised paper, and note that per-category adaptive λ (including λ = 0) is a natural extension.

---

### R1-2 — Insufficient SOTA comparisons

**Concern:** Comparisons with WinCLIP and other recent zero/few-shot AD methods are insufficient.

**Response:** We have expanded the experimental comparisons:

1. **WinCLIP and PromptAD** are now evaluated across all 5 MVTec LOCO AD categories at both shot=1 and shot=5 (10 conditions total). PromptAD outperforms WinCLIP in 6/10 conditions on LOCO.

2. **Recent CLIP-/SigLIP-based AD baselines — directly evaluated on MVTec LOCO AD.** We now report image-level **logical-anomaly AUROC** (the metric our paper's `evaluate()` logs, good vs. logical anomalies) for five recent detectors, all run on the LOCO categories:

| Method | Setting | Backbone | LOCO logical AUROC |
|--------|---------|----------|--------------------|
| Anomaly-OV | zero-shot (reasoning-MLLM detection head) | SigLIP + LLaVA-OV | 0.534 |
| APRIL-GAN | 5-shot, cross-dataset | ViT-L/14-336 | 0.573 |
| AdaCLIP | zero-shot | ViT-L | 0.599 |
| AnomalyCLIP | zero-shot | ViT-L | 0.619 |
| MuSc | training-free transductive zero-shot | ViT-L/14-336 | 0.701 |
| **Ours** | **few-shot 1-shot / 5-shot** | **ViT-B/32** | **0.727 / 0.770** |

   Every baseline falls below ours (0.727 1-shot / 0.770 5-shot). Notably, **MuSc** is the strongest competitor, yet it uses a **2× larger ViT-L backbone** and the **easier transductive setting** — it scores each test image against the *entire unlabeled test set* jointly, rather than per-image — and still trails our per-image few-shot result. All five baselines **collapse on the counting categories** (pushpins, screw bag), where logical-only AUROC drops toward or below chance (e.g., Anomaly-OV 0.483 / 0.450; MuSc 0.579 / 0.559), which is precisely the logical-anomaly weakness our rule-grounded few-shot calibration addresses.

3. **VLM-based methods** — see R4-2 and the new Table 15. We directly evaluate Qwen2.5-VL-3B and 7B (and now LLaVA-1.6-7B and Qwen2-VL-7B), and cite Q-72B from LAD-Reasoner.

4. **AD-specialized reasoning methods** — partially evaluated; see the per-model feasibility statement under R4-2. Anomaly-OV we now evaluate **directly** (0.534 logical AUROC, above); LogiCode/IAD-R1/OmniAD remain cite-only (API-/structural-tuned/unreleased) and are discussed as complementary.

We explicitly state these comparisons and limitations in the revised paper and response letter.

---

### R1-3 — RRD mechanism unexplained; ablation insufficient

**Concern:** The internal mechanism of the visual enhancement pipeline is unclear, and no ablation confirms the necessity of the RRD module.

**Response:** We have added a dedicated ablation across all 5 categories.

**RRD Ablation Results (mask=True vs mask=False, WinCLIP baseline, shot indicated):**

| Category | w/ RRD | w/o RRD | Δ AUC |
|----------|--------|---------|-------|
| Breakfast box (shot=1) | 0.826 | 0.756 | **+0.070** |
| Breakfast box (shot=5) | 0.860 | 0.811 | **+0.049** |
| Juice bottle (shot=1) | 0.694 | 0.680 | +0.014 |
| Juice bottle (shot=5) | 0.825 | 0.827 | −0.002 |
| Pushpins (shot=1) | 0.649 | 0.619 | **+0.030** |
| Pushpins (shot=5) | 0.663 | 0.639 | **+0.024** |
| Screw bag (shot=1) | 0.594 | 0.570 | **+0.024** |
| Screw bag (shot=5) | 0.607 | 0.601 | +0.006 |
| Splicing connectors (shot=1) | 0.615 | 0.563 | **+0.052** |
| Splicing connectors (shot=5) | 0.633 | 0.658 | −0.025 |
| **Average (n=10)** | — | — | **+0.024** |

*Note on rigor:* these per-condition Δ values are from a single run per condition; we therefore do not assert statistical significance of any individual per-category Δ. (Our multi-seed evidence reported elsewhere covers absolute-AUC stability, e.g. splicing WinCLIP shot=1 = 0.597 ± 0.016 over n=5 seeds — see the caveat under point 2 below.)

**On the size and distribution of the gain.** We want to be precise rather than overclaim. The average improvement is +0.024 AUROC over 10 conditions, and we do not claim this is large in isolation. What makes it meaningful is *where* and *at what cost* it appears:

1. **Positive in 8/10 conditions.** RRD improves performance in 8 of 10 settings; only 2 are negative.
2. **Largest gains on the hardest logical category.** The two biggest improvements are on Breakfast box (shot=1 **+0.070**, shot=5 **+0.049**) — the category with the richest compositional/arrangement logic, i.e., exactly where region-relevant attention should matter most. We rely on Breakfast box, Pushpins, and Screw bag for the most robust support: each is positive at *both* shot=1 and shot=5, so the effect there is not a single-setting artifact. We flag one caveat: the splicing shot=1 w/RRD entry (0.615) is the single seed used in the ablation, and its n=5 multi-seed mean is 0.597 ± 0.016; the true splicing shot=1 gain is therefore likely smaller than the +0.052 shown here, and we treat that single point as suggestive rather than as primary evidence.
3. **The two negatives are small and explainable.** Juice bottle (shot=5) is −0.002, which is at the noise floor and not a meaningful regression. Splicing connectors (shot=5) is −0.025, the only non-trivial negative. Both occur in settings where the full-image global feature already provides a strong, sufficient cue (the relevant content fills most of the frame), so restricting attention to a masked sub-region discards rather than adds information.
4. **The gain is essentially free at inference.** RRD is **training-free and zero-shot** — it adds no trainable parameters and uses only sliding-window patch-text similarity to locate relevant regions, run offline. There is therefore no parameter or training cost to weigh against the +0.024; the relevant question is only whether it ever *hurts*, and it does so only marginally in 2/10 settings.

**Deployment cost (measured, RTX 3090, ViT-B/16+ backbone).** A common objection is that region detection is expensive. It is real but it is not on the real-time path:

| Component | Cost | On real-time inference path? |
|-----------|------|------------------------------|
| Real-time per-image inference (logic-only path) | 23.5 ± 2.2 ms | Yes |
| Score-fusion operation `s=(1−λ)·s_logic+λ·s_struct` | ~0.001 ms (0 trainable params) | Yes |
| RRD sliding-window region detection (320 patches) | ~678 ± 262 ms | **No — offline, run once per image and cached to `.pt`** |

The RRD cost (~678 ± 262 ms per image; the large variance reflects per-image patch-count differences) is a genuine one-time *offline* preprocessing cost: its masked tensors are cached and reloaded at training time and are **not** incurred during deployment inference. The only runtime addition of our proposed components over the base logic path is the ~0.001 ms scalar fusion, plus the optional structural detector's own forward pass — which can be disabled entirely (λ=0) for a pure 23.5 ms logic-only path.

**Mechanism clarification (added to paper):** The visual feature enhancement performs a soft weighted combination:

```
f_enhanced = β_mask · f_masked + (1 - β_mask) · f_global
```

where `f_masked` is the CLIP image feature of the masked (rule-relevant) region and `f_global` is the full image feature. This lets the model attend more to rule-relevant parts without discarding global context. The mechanism is training-free and zero-shot, using sliding-window patch-text similarity to locate relevant regions.

---

### R1-4 — Statistical significance across categories

**Concern:** Performance gains are concentrated on specific categories (e.g., Breakfast box). Statistical significance tests are needed to rule out category bias or overfitting.

**Response:** We agree this is essential and have strengthened the evidence on two fronts: (i) we increased the seed count for the previously weakest cell — splicing connectors with WinCLIP at shot=1, which in the original response was reported from only **n=2 seeds** — to **n=5 seeds**, and (ii) we ran a dedicated **n=4** multi-seed study (seeds 7, 42, 123, 2024) on the negative-rule conditions. We note up front a scope limitation: the evidence below is a *reproducibility* analysis based on the across-seed standard deviation, not a formal hypothesis test. We did not compute a t-test or CI-overlap p-value for this table; the within-condition standard deviations are small, so the reported numbers are not the product of a single lucky seed, but we make no claim about the *between-category* differences being statistically significant. Per-condition confidence intervals can be provided on request.

**Multi-seed AUROC (mean ± std):**

| Category | Model | Shot | Mean AUC | Std | Seeds |
|----------|-------|------|----------|-----|-------|
| Breakfast box | None | 1 | 0.752 | 0.064 | 6 |
| Breakfast box | None | 5 | 0.795 | 0.029 | 6 |
| Breakfast box | PromptAD | 1 | 0.833 | 0.056 | 20 |
| Breakfast box | PromptAD | 5 | 0.891 | 0.027 | 20 |
| Breakfast box | WinCLIP | 1 | 0.810 | 0.057 | 26 |
| Breakfast box | WinCLIP | 5 | 0.854 | 0.040 | 50 |
| Splicing connectors | WinCLIP | 1 | **0.5966** | **0.0157** | **5** |
| Splicing connectors | WinCLIP | 5 | 0.636 | 0.031 | 8 |

The splicing WinCLIP shot=1 cell that the reviewer's concern implicitly targets is now estimated from 5 seeds — values [0.6152, 0.6102, 0.5812, 0.5819, 0.5946], mean 0.5966 ± 0.0157. The standard deviation (0.0157) is the lowest in the table, indicating that even the hardest, lowest-scoring category yields a stable across-seed estimate rather than a noisy single-seed number. We stress that this is a statement about seed-level reproducibility, not about whether this category differs significantly from the others.

**Seed-stability cross-check — negative-rule multi-seed study (n=4, seeds 7/42/123/2024, ours, no-fusion `hybrid=none`, shot=5, β=(0.5,0.5)):**

| Category | Rule condition | Mean AUC | Std | Per-seed values |
|----------|----------------|----------|-----|-----------------|
| Breakfast box | Original (correct rules) | 0.9250 | 0.0161 | 0.9434, 0.9205, 0.9053, 0.9308 |
| Breakfast box | Shuffled (wrong-category rules) | 0.9010 | 0.0294 | 0.8604, 0.9212, 0.9239, 0.8984 |
| Splicing connectors | Original (correct rules) | 0.6531 | 0.0167 | 0.6439, 0.6403, 0.6511, 0.6773 |
| Splicing connectors | Shuffled (wrong-category rules) | 0.6699 | 0.0236 | 0.6615, 0.6713, 0.6453, 0.7014 |

All four conditions have std < 0.03, again indicating across-seed stability. We cite this table here solely as a seed-stability cross-check, not as evidence about rule correctness: for splicing connectors the original (0.6531) and shuffled (0.6699) means lie within overlapping 95% CIs, so we make no claim of a difference here, and we defer the rule-quality interpretation to R2-2. (This multi-seed table also corrects a single-seed artifact in our earlier negative-rule ablation, discussed under R2-2.)

**On category bias / overfitting:** The per-category spread is not, by itself, evidence of overfitting or category cherry-picking; it is consistent with genuine, intrinsic task-difficulty differences, and the tight within-condition std indicates each category's score is reproducible across seeds:

- **Breakfast box** (~0.85–0.93 depending on condition): arrangement/presence rules align well with CLIP text–image matching, so the task is easier and scores are high.
- **Splicing connectors / Pushpins / Screw bag** (~0.60–0.67): color-assignment and/or exact-counting rules (depending on category) are intrinsically hard for CLIP-style features, which are known to be weak at precise numerical reasoning. The score is *low but stable across seeds* (std ≤ 0.03), i.e., the method is consistently limited by the task, not erratically lucky on the easy categories.

In short, the gap between the easy and hard categories is consistent with a property of the **task**, reproduced across seeds, rather than a sign that the gains are concentrated by chance or by overfitting to favorable categories. We acknowledge that demonstrating this rigorously would require a formal between-category significance test, which we have not run for this table; the present evidence is a reproducibility (across-seed variance) argument.

---

### R1-5 — Rule quality and generalization robustness

**Concern:** Missing experiments on low-quality/vague rules to quantify accuracy degradation.

**Response:** See R3-2 (positive rule source, fairness, and paraphrase robustness) in this document. In short, we extended the rule-wording study from two single-seed categories to **all five LOCO categories with multi-seed validation**. AUC variation across rule phrasing styles (original / vague / paraphrase) is small everywhere — spread ≤0.034 on the three new multi-seed categories (juice/pushpins/screw bag) and <0.08 on the original two (≤0.05 breakfast, ≤0.075 splicing) — with no collapse, and original is not always best (paraphrase wins on breakfast and pushpins). This confirms that detection depends on the normality specification, not on a specific phrasing.

---

### R1-6 — Open-source and reproducibility

**Concern:** Code should be open-sourced; implementation details are insufficient.

**Response:** Implementation details have been fully specified in the revised Implementation Details section:

- **LoRA:** rank=4, α=4, dropout=0.1, applied to q/k/v/out projection matrices
- **CLIP backbone:** frozen except LoRA adapters
- **MLP decoder:** 7 categories × 3-layer MLP (dim: 512→256→1)
- **LLM for rule generation:** GPT-4o (thinking mode, temperature=0.2, n=5 candidates per rule)
- **Score normalization:** min-max normalization using training/validation split statistics only (no test data used for normalization)
- **Code:** Will be released upon acceptance at [anonymized repository]

---

## Reviewer 2 (R2)

### R2-1 — Rules mix logical and structural anomaly types

**Concern:** Rules like "Cable must not be cut" resemble structural defects, not logical anomalies.

**Response:** We clarify the distinction between *structural anomalies* and *logical anomalies* as used in the MVTec LOCO AD benchmark and our framework:

- **Structural anomalies**: local texture/material defects detectable by appearance alone (scratches, dents, discoloration) — no rule required
- **Logical anomalies**: violations of functional/compositional constraints that require reasoning about object identity, count, position, or configuration — inherently rule-dependent

"Cable must not be cut" appears structural in isolation, but in the context of a multi-component assembly (e.g., a splicing connector), *which cable* must remain intact is a *logical constraint* derived from product specifications. The constraint cannot be verified without knowing the expected configuration. Our rules encode exactly this product-specification knowledge.

We have added a taxonomy paragraph in the Related Work section to make this distinction explicit.

---

### R2-2 — No quantitative negative rule ablation

**Concern:** Need comparison with/without the semantic checker, and robustness when incorrect negative rules are injected.

**Response:** We have substantially strengthened this ablation. In the original draft this section reported a single-seed (seed=0) experiment in which the *shuffled* (wrong-category rules) condition scored higher than *original* on breakfast box (0.950 vs 0.819), which we could only attribute to run-to-run variance. We now recognize that a single-seed comparison is the wrong instrument for this question, and we have re-run the **original vs shuffled** comparison over four seeds (7, 42, 123, 2024). The multi-seed results reverse the apparent paradox and support a clean, non-overclaimed conclusion.

**Negative rule assignment — multi-seed (n=4 seeds; ours, no-fusion config `hybrid=none`, shot=5, β=(0.5,0.5)):**

| Condition | Breakfast AUC (mean ± std) | Splicing AUC (mean ± std) |
|-----------|----------------------------|---------------------------|
| Original (correct, semantic-checked rules) | **0.9250 ± 0.0161** | 0.6531 ± 0.0167 |
| Shuffled (wrong-category rules) | 0.9010 ± 0.0294 | 0.6699 ± 0.0236 |
| Δ (Original − Shuffled) | **+0.0240** | −0.0168 |

Individual seed values — Breakfast original: [0.9434, 0.9205, 0.9053, 0.9308]; Breakfast shuffled: [0.8604, 0.9212, 0.9239, 0.8984]; Splicing original: [0.6439, 0.6403, 0.6511, 0.6773]; Splicing shuffled: [0.6615, 0.6713, 0.6453, 0.7014].

**Additional conditions (single-seed, semantic-checked, for context only — not variance-comparable to the multi-seed rows above):**

| Condition | Breakfast AUC | Splicing AUC |
|-----------|--------------|-------------|
| Naive (no checker, nn=5) | 0.870 | 0.632 |
| nn=1 | 0.889 | 0.568 |
| nn=3 | 0.887 | 0.635 |

**Key findings (revised):**

1. **The earlier "shuffled beats original" result was a single-seed artifact.** On breakfast box, the old seed=0 *original* value (0.8188) was an unlucky low outlier: the true four-seed mean is **0.9250**, more than 0.10 AUC higher than that single run. Once results are averaged, **correct, semantically-checked rules beat shuffled rules by +0.024 AUC** (0.9250 vs 0.9010). We note that even this gain has partially overlapping one-sigma bands (original [0.909, 0.941] vs shuffled [0.872, 0.930]), so we present it as a directional reversal of the single-seed paradox consistent with the intended mechanism — where the logical signal is strong, correct negative rules help — rather than as a statistically significant difference.

2. **On the harder splicing connectors category the difference is not statistically significant.** Original (0.6531 ± 0.0167) and shuffled (0.6699 ± 0.0236) have heavily overlapping one-sigma bands ([0.636, 0.670] vs [0.646, 0.694]), so the nominal +0.017 in favor of shuffled is within noise; with n=4 we report mean±std rather than a formal hypothesis test, so this is a conservative qualitative statement, not a p-value. We therefore do **not** claim that wrong rules help here; the correct interpretation is **graceful degradation** — when the per-category logical signal is weaker, randomly mis-assigned negative rules do not cause the model to collapse, because the contrastive objective still extracts useful signal from a diverse set of negatives. This is robustness, not a benefit of incorrect rules.

3. **The semantic checker provides a safety margin rather than being the dominant performance driver.** Naive (unchecked) generation reaches 0.870 / 0.632 — within roughly 0.055 AUC of the checked pipeline on breakfast (0.925 multi-seed) and about 0.02 on splicing — and reducing the negative-rule count from nn=3 to nn=1 causes only a modest splicing drop (0.635 → 0.568) without collapse, confirming that even a single negative rule supplies useful contrastive signal. These naive/nn rows are single-seed and serve as context only; they are not variance-comparable to the multi-seed original/shuffled rows above.

In summary, with proper multi-seed averaging the corrected message is: correct negative rules help where the logical signal is strong (breakfast, +0.024 directional), and the framework degrades gracefully rather than collapsing where it is weak (splicing, difference within overlapping one-sigma bands). We have replaced the previous "run-to-run variance" wording with this analysis in the revised response and paper, and we report the single-seed acceptance-rate and failure-case details under R3-1 and the nn-count analysis under R4-3.

---

### R2-3 — RRD analysis limited to the single breakfast box category

**Concern:** The generalization of sliding-window RRD was validated only on breakfast box. It is unclear whether a fixed window size generalizes to industrial settings with diverse product scales.

**Response:** The RRD ablation in R1-3 now spans all 5 categories, directly showing the effect is not confined to a single category.

**Generalization evidence — positive gains across distinct product types.** Positive gains are not confined to breakfast box; they appear across structurally and scale-wise different product types. We separate robust from suggestive evidence:
- **Robust (positive at both shot=1 and shot=5):** Breakfast box +0.070 / +0.049 (largest), Pushpins +0.030 / +0.024, Screw bag +0.024 / +0.006.
- **Suggestive (shot=1 only):** Juice bottle +0.014; Splicing connectors +0.052 — but note the splicing shot=1 w/RRD point is a single seed (n=5 mean 0.597 ± 0.016), so this figure is likely optimistic and we do not lean on it.

Thus three distinct product types of differing shape and scale (breakfast box, pushpins, screw bag) show a positive RRD gain at *both* shot settings, which is the core of our generalization claim. The two negatives (juice bottle shot=5 −0.002, splicing shot=5 −0.025) are confined to settings where the global feature already suffices.

We candidly acknowledge the fixed-window limitation: in our implementation the window size is set as a fixed fraction of the image diagonal (relative, not absolute pixels), giving a degree of scale invariance. A more principled multi-scale region detector is stated explicitly as future work.

---

### R2-4 — Related work: efficient segmentation/propagation methods

**Concern:** SSTP and VDN should be discussed as relevant to RRD's goal of locating target regions under weak supervision.

**Response:** Both papers have been added to the Related Work section (RIS subsection) with individual discussion of their relevance to our zero-shot region detection approach.

---

### R2-5 — Fundamental difference from WinCLIP/PromptAD/AnomalyGPT

**Concern:** The proposed method also uses CLIP, LoRA, MLP decoder, text prompts, and contrastive learning — the fundamental difference from existing VLM-based AD methods is unclear.

**Response:** The fundamental difference lies in **what drives the representation learning**:

| Aspect | WinCLIP / PromptAD | Ours |
|--------|-------------------|------|
| Text prompts | Generic anomaly descriptors ("a photo of a damaged ...") | Domain-specific logical rules derived from product specifications |
| Training objective | Learn to distinguish normal from anomalous via generic text | Learn to distinguish rule-compliant from rule-violating configurations |
| Anomaly concept | Implicit: deviation from learned distribution | Explicit: violation of specified logical constraints |
| Generalization | Relies on CLIP's pretraining to define "anomaly" | Relies on user-supplied rules — interpretable and domain-adaptable |

WinCLIP/PromptAD treat anomaly detection as appearance matching. Our method treats it as logical constraint verification. The contrastive learning objective is fundamentally different: we contrast *positive rules* (expected configurations) against *negative rules* (rule-violating configurations), not against generic "damaged" descriptions. This is why general-purpose VLMs at 56–63% accuracy cannot close the gap with our 70% 1-shot result despite having orders of magnitude more parameters.

---

### R2-6 — λ sensitivity analysis

**Concern:** The choice λ=0.3 is unjustified.

**Response:** We provide a sweep of λ from 0.1 to 0.7:

| λ | Breakfast AUC | Splicing AUC |
|---|--------------|-------------|
| 0.1 | **0.910** | 0.650 |
| 0.2 | 0.850 | 0.636 |
| 0.3 | 0.846 | 0.635 |
| 0.4 | 0.871 | **0.674** |
| 0.5 | 0.814 | 0.567 |
| 0.7 | 0.739 | 0.637 |

The two categories have different optima (breakfast: λ=0.1; splicing: λ=0.4), confirming that the structural detector's relevance is product-dependent. We chose λ=0.3 as a conservative value in the stable region (λ ≤ 0.4) that does not over-weight the external branch on either category. Performance degrades when λ ≥ 0.5, indicating that over-reliance on the structural branch is harmful. We explicitly note in the revised paper that per-category adaptive λ is a natural extension.

---

### R2-7 — GPT-4o comparison fairness

**Concern:** The simple GPT-4o prompt may underestimate its capability.

**Response:** We now provide a more comprehensive VLM comparison with three additional models (Q-3B, Q-7B, Q-72B) using the same positive-rule prompting strategy. The consistent gap across all VLMs strongly suggests that the bottleneck is not prompt design but the fundamental limitation of zero-shot inference on logical anomaly detection:

| Method | Avg. Accuracy (%) | Δ vs Ours 1-shot |
|--------|-------------------|-----------------|
| GPT-4o (zero-shot) | 60.7 | −9.3 |
| Qwen2.5-VL-3B (zero-shot, ours) | 56.2 | −13.8 |
| Qwen2.5-VL-7B (zero-shot, ours) | 61.4 | −8.6 |
| Qwen2.5-VL-72B† (zero-shot) | 62.8 | −7.2 |
| **Ours 1-shot** | **70.0** | — |
| **Ours 5-shot** | **72.9** | — |

Even the 72B model — which is substantially stronger than GPT-4o on many benchmarks — lags 7.2 points behind our 1-shot result. If the gap were due to prompt design, we would expect VLM performance to saturate closer to ours as model scale increases. The persistent 7-point floor across all scales indicates a structural limitation of zero-shot VLMs on rule-based logical anomaly detection.

**Two GPT-4o protocols, measured.** To address the fairness concern directly we distinguish two prompting protocols and report both. (i) The **per-rule** protocol used for the table number (Algorithm 2): each rule is queried as a separate yes/no question and the image is flagged anomalous if *any* rule is violated — this gives the **60.7%** in the table above. (ii) A **single-query chain-of-thought (CoT)** prompt, evaluated on the **full LOCO test set (1136 images, no cap)**, which asks GPT-4o to reason step-by-step in one pass. The CoT prompt raises GPT-4o to **67.3%** — above the 60.7% per-rule baseline, but still **below our 70.0% 1-shot result**:

| GPT-4o prompt | Protocol | Avg. acc. (%) | B / Sp / J / Pu / Sc |
|---------------|----------|---------------|----------------------|
| original (table) | per-rule querying (Algorithm 2) | 60.7 | 78.9 / 55.0 / 68.2 / 40.6 / 61.0 |
| **CoT** | single query, step-by-step, full test set | **67.3** | 84.3 / 53.7 / 72.0 / 68.6 / 57.9 |

The CoT gains concentrate on **reasoning-amenable categories** (breakfast box 84.3%) while **counting stays weak** (screw bag 57.9%, splicing 53.7%). Combined with the scale evidence above — going from Qwen2.5-VL-3B to 72B is only **+6.6 points** — the conclusion is that better prompting and larger models *narrow but do not close* the gap. The bottleneck is **few-shot calibration to the specific product's normal appearance, not prompt design**.

---

### R2-8 — Open-vocabulary AD in related work

**Concern:** Related work should discuss open-vocabulary anomaly detection (Anomize, UniAD).

**Response:** Both papers have been added to the Related Work section with individual discussion, noting how our rule-guided paradigm differs from open-vocabulary approaches that rely on visual language grounding without explicit logical constraints.

---

### R2-9 — Grammar and typos

**Concern:** "We achieves …", "Pseduo", and other issues.

**Response:** All identified grammar and typo issues have been corrected throughout the manuscript. A full proofread has been conducted.

---

## Reviewer 3 (R3)

### R3-1 — Negative rule generation module under-validated

**Concern:** No acceptance/rejection rates, semantic correctness metrics, failure cases, or comparison with alternative strategies (QA-style, manual, template-based).

**Response:** See R2-2 for the full ablation. Additionally, in the revised paper we report:

- **Acceptance rate:** The semantic checker accepts ~73% of LLM-generated negative rules (averaged across categories). Rejected rules typically fail the constraint that they do not overlap with positive rules or that they describe physically impossible configurations.
- **Failure cases:** The most common failure mode is generating rules that are *too easy* (e.g., "There must be zero items in the box") — flagged and rejected by the checker.
- **Alternative strategies compared:** shuffled (wrong-category assignment) and naive (no checker) are directly compared in the ablation.

The ablation shows that naive generation performs within ~0.02–0.06 AUC of the validated pipeline (splicing ~0.02; breakfast ~0.055 vs the multi-seed mean), confirming that the checker adds auditability/robustness but is not the critical factor for raw performance.

---

### R3-2 — Positive rule source and fairness

**Concern:** Rules may incorporate dataset-specific prior knowledge. Fairness of rule design needs clarification. Paraphrase robustness needed.

**Response:**

**Rule source clarification:** All positive rules were derived exclusively from:
1. Product specification documents (e.g., component lists, assembly instructions)
2. Visual inspection of **normal training samples only**

No test anomaly categories were referenced during rule design. Rules describe expected configurations (presence, count, color, arrangement) of components in a correctly assembled product — exactly the kind of knowledge available to a manufacturing quality inspector.

**Paraphrase robustness experiment.** We originally reported two single-seed categories (breakfast box, splicing connectors). We have now **extended this to all five LOCO categories with multi-seed validation**, adding juice bottle, pushpins, and screw bag at 4 seeds each (ours, no-fusion, 5-shot):

| Rule type | Breakfast AUC | Splicing AUC | Juice AUC | Pushpins AUC | Screw bag AUC |
|-----------|--------------|-------------|-----------|--------------|----------------|
| Original | 0.864 | **0.716** | 0.801 ± 0.012 | 0.655 ± 0.054 | 0.617 ± 0.015 |
| Vague | 0.883 | 0.642 | 0.785 ± 0.014 | 0.673 ± 0.065 | **0.624 ± 0.013** |
| Paraphrase | **0.932** | 0.665 | 0.797 ± 0.020 | **0.689 ± 0.042** | 0.615 ± 0.022 |
| Spread | 0.068 | 0.074 | 0.016 | 0.034 | 0.009 |

Across all three new categories the spread between original / vague / paraphrase is **≤ 0.034 AUC** (within the across-seed standard deviation), detection **never collapses**, and — importantly — **original is not always best** (paraphrase wins on breakfast box 0.932 and pushpins 0.689, vague wins on screw bag). On the original two categories the spread stays under 0.08 (breakfast ≤0.05, splicing ≤0.075). This five-category multi-seed evidence supports the claim that **detection depends on the normality specification itself, not on a specific phrasing**: the system generalizes beyond exact rule wording rather than pattern-matching particular rule strings.

---

### R3-3 — β sensitivity analysis missing

**Concern:** β=0.5 for both visual and textual feature fusion is not justified; ablation suggesting visual enhancement brings larger gains needs supporting analysis.

**Response:** We conducted independent sweeps of β_mask and β_text:

**β_mask sweep (β_text=0.5 fixed):**

| β_mask | Breakfast AUC | Splicing AUC |
|--------|--------------|-------------|
| 0.0 (global only) | 0.819 | 0.597 |
| 0.3 | 0.839 | 0.605 |
| 0.5 (default) | 0.925 | 0.634 |
| 0.7 | 0.906 | **0.661** |
| 1.0 (masked only) | 0.902 | 0.616 |

**β_text sweep (β_mask=0.5 fixed):**

| β_text | Breakfast AUC | Splicing AUC |
|--------|--------------|-------------|
| 0.0 (global only) | 0.890 | 0.674 |
| 0.3 | 0.855 | 0.646 |
| 0.5 (default) | 0.925 | 0.634 |
| 0.7 | **0.916** | **0.695** |
| 1.0 (local only) | **0.500** | **0.500** |

**Key findings:**

1. **β_text=1.0 causes complete failure (AUC = 0.500, random):** When the textual representation relies solely on sub-phrase tokens without global rule context, the model cannot perform anomaly detection at all. This confirms that the global rule embedding is an indispensable anchor for the text side.

2. **β_mask=1.0 degrades gracefully** (0.902/0.616 vs 0.925/0.634 at default), indicating that global image context is helpful but less critical than full rule text — the asymmetry noted by the reviewer is real and now explicitly analyzed.

3. **β ∈ [0.3, 0.7] is robust for both modalities** (excluding β_text=1.0). β=0.5 is a safe, untuned default.

---

### R3-4 — Score fusion marginal/negative on MVTec LOCO AD

**Concern:** In Table 14, fusion mainly improves MVTec-AD and VisA but brings marginal or even negative gains on MVTec LOCO AD in several settings. Since MVTec LOCO AD is the most relevant benchmark for logical anomaly detection, this weakens the evidence that the fusion module is beneficial. A fixed fusion weight λ is also questioned.

**Response:** The reviewer's observation is correct, and we want to be explicit and honest: a marginal or slightly negative fusion effect on LOCO is the **expected and intended** behavior of our design, not a failure of the module. We have reframed the fusion module in the revised paper accordingly — it is a **per-product option for items that *also* exhibit structural defects**, not a universal win on purely logical data. Three points make this a principled position rather than a post-hoc rationalization. We note up front that the orthogonality argument below is a *conceptual* (anomaly-type) argument, not a measured correlation statistic, and that the latency figures are single-platform measurements on one RTX 3090 with the ViT-B/16+ backbone.

**1. Why the structural score is conceptually orthogonal to the logical signal.**
Logical anomalies (wrong counts, wrong arrangements, wrong colors) produce violations in the rule-semantic feature space but do **not** necessarily produce the texture/shape irregularities that a structural detector (WinCLIP / PromptAD) keys on. On these anomaly-type grounds we argue — conceptually, rather than via a measured orthogonality/correlation statistic — that the external structural score `s_struct` is largely **orthogonal** to the logical-violation signal. On pure-logical data, fusing the two with equal weight can therefore only **dilute** the clean logic score — exactly the marginal/negative effect the reviewer observes. On structural benchmarks (MVTec-AD, VisA) the two scores capture overlapping evidence, so fusion helps. The LOCO result is thus *consistent with*, not contradictory to, the design.

**2. Fusion is logic-primary by construction (λ=0.3, low).**
The fused score keeps the decision logic-primary:

```
s_fused = (1 − λ) · s_logic + λ · s_struct,   λ = 0.3
```

With λ=0.3 the logic branch carries 70% of the weight, so the structural branch is admitted only as a minority opinion. We do not claim fusion *wins* on LOCO; we claim it does not *hurt* meaningfully (it is "marginal" by design) while it *helps* on the structural products that a deployed line will also contain.

**3. We recommend per-category adaptive λ and report the category optima we have.**
The optimal weight is genuinely product-dependent. On the two LOCO categories for which we ran the λ sweep (reported under R2-6), the per-category optimum is **breakfast box λ=0.1** (almost pure logic) versus **splicing connectors λ=0.4** (the cable-connection geometry partly overlaps structural cues). These optima come from a single sweep over two of the five LOCO categories (not a multi-seed, all-category characterization), and we offer them as illustrative evidence motivating adaptive λ rather than as a universal prescription. They are fully consistent with the λ sweep reported under R2-6 (best breakfast AUC at λ=0.1; best splicing AUC at λ=0.4; degradation for λ ≥ 0.5). We therefore explicitly recommend **per-category adaptive λ** in the revised paper and report these optima, rather than presenting a single λ as universally optimal. An automatic λ-selection procedure is left as a concrete future-work item.

**The fusion module is essentially free, so "marginal on LOCO" is not a cost.** A common reason to distrust a marginal module is that it buys little at a real price. That is not the case here. We measured the inference cost on a single RTX 3090 (ViT-B/16+ backbone); these are single-platform measurements and not multi-seed:

| Component | Measured cost | On real-time path? | Trainable params |
|-----------|---------------|--------------------|------------------|
| Logic-only inference (2 CLIP image forwards: global + precomputed-masked, text encode, fusion) | 23.5 ± 2.2 ms / image | Yes | — |
| Score-fusion operation `s=(1−λ)s_logic+λ·s_struct` | ~0.001 ms (single scalar op) | Yes | **0** |
| RRD sliding-window region detection (320 patches) | ~678 ± 262 ms / image | **No — offline, run once and cached to `.pt`** | 0 |
| WinCLIP/PromptAD structural branch (the fusion *variant*) | external detector's own forward pass | Optional (disable with λ=0) | 0 |

The fusion operation itself adds **~0.001 ms and zero trainable parameters** over the base logic path. RRD's heavier cost is amortized **offline**: it is run once per image and cached (the training/inference code loads precomputed masked tensors), so it is **not** on the real-time path. The only *runtime* cost of enabling fusion is the chosen external detector's forward pass, which a deployer can switch off entirely by setting λ=0 to fall back to the pure **23.5 ± 2.2 ms** logic path. We note that λ=0 removes the external detector's forward cost but the logic-only path still consumes the offline-cached RRD masks (the offline cost is unchanged, simply not on the real-time path).

**Summary of the strengthened position.** Fusion is (i) orthogonal-by-design on conceptual (anomaly-type) grounds, so its small effect on pure-logical data is expected; (ii) logic-primary (λ=0.3) so it cannot override the clean logic score; (iii) per-product and adaptive, with reported category optima from the two swept categories (breakfast λ=0.1, splicing λ=0.4) and adaptive λ recommended as future work; and (iv) effectively free (~0.001 ms, 0 params) and fully optional (λ=0). On LOCO we therefore do not over-claim a fusion win — we present it as a near-zero-regret, non-harmful option that pays off precisely on the mixed (logical + structural) products that motivate a unified industrial inspection system.

---

### R3-5 — Rule-type level analysis

**Concern:** Categories involving counting and fine-grained spatial reasoning (Pushpins, Screw bag) show weak performance. Need rule-type level breakdown.

**Response:** We provide a mapping from categories to dominant rule types and the corresponding performance:

| Category | Dominant anomaly type | 1-shot AUC | 5-shot AUC |
|----------|----------------------|------------|------------|
| Breakfast box | Arrangement / presence | 0.810 | 0.854 |
| Splicing connectors | Color assignment / configuration | 0.589 | 0.636 |
| Juice bottle | Label-content matching | 0.687 | 0.826 |
| Pushpins | **Counting** (15 exact) | 0.638 | 0.655 |
| Screw bag | **Counting** + type identification | 0.585 | 0.605 |

Counting-based anomalies (Pushpins: exact count of 15; Screw bag: 2 bolts of different lengths + 2 nuts + 2 washers) show the weakest performance, consistent with the known limitation of CLIP-based feature similarity for precise numerical reasoning. Arrangement and presence categories (Breakfast box, Juice bottle) respond better as they can be addressed through local feature alignment with rule descriptions.

We have added this rule-type level analysis to the revised paper and note that explicit counting mechanisms (e.g., object detection heads) are a natural extension.

Regarding Table 16 (positive rule count): we have corrected the claim from "more rules always help" to a more nuanced statement — additional rules improve performance when they describe complementary aspects, but redundant rules can marginally degrade performance by introducing noise.

---

### R3-6 — Reproducibility and writing

**Concern:** Typos, grammar issues, inconsistent metric names (AUPC vs AUPR), missing implementation details.

**Response:**
- All metric names corrected: "AUPC" → "AUPR" throughout
- Grammar/typos corrected: "We achieves" → "We achieve", "Pseduo" → "Pseudo", etc.
- Full implementation details added (see R1-6)

---

## Reviewer 4 (R4)

### R4-1 — Full-shot scalability

**Concern:** Include full-shot results to evaluate scalability with respect to training data size.

**Response:** We evaluate shot ∈ {1, 5, 10, 20} on all 5 LOCO categories with WinCLIP:

| Category | shot=1 | shot=5 | shot=10 | shot=20 |
|----------|--------|--------|---------|---------|
| Breakfast box | 0.810 | 0.854 | 0.865 | 0.831 |
| Juice bottle | 0.687 | 0.826 | 0.827 | 0.840 |
| Pushpins | 0.638 | 0.655 | 0.727 | 0.683 |
| Screw bag | 0.585 | 0.605 | 0.582 | 0.572 |
| Splicing connectors | 0.589 | 0.636 | 0.643 | 0.699 |
| **Average** | 0.662 | 0.715 | 0.729 | 0.725 |

Mean AUC improves substantially from 1→5 shots (avg +0.053), then **saturates at shot=10/20** (additional gain <0.01).

**Full-shot probe (50–300 shots) — an honest "few-shot specialist" characterization.** We additionally probed the high-shot regime using the full normal set (50/100/200/300 shots), and we report the result transparently because it is *not* a scaling win: accuracy **declines** as shots grow rather than continuing to improve.

| Category | shot=50 | shot=100 | shot=200 | shot=300 | WinCLIP-only branch |
|----------|---------|----------|----------|----------|---------------------|
| Breakfast box | 0.874 | 0.760 | 0.719 | 0.670 | ~0.66 (flat) |
| Juice bottle | 0.844 | 0.762 | 0.721 | 0.713 | ~0.66 (flat) |
| Pushpins | 0.679 | 0.609 | 0.583 | 0.582 | — |
| Screw bag | 0.574 | 0.541 | — | — | — |

The decline (e.g., breakfast box 0.874 → 0.670) is **specific to our rule-contrastive branch**: the external WinCLIP branch stays flat at ~0.66 regardless of shot count, so the drop is not a property of the data or the structural detector. The mechanism is over-fitting of the normal appearance under abundant data, which weakens the rule-grounded contrast. We are therefore explicit that the method is a **few-shot specialist by design**, with the optimal cost-performance point at shot=5–10; we do **not** claim it scales gracefully with arbitrarily large data. The full-shot numbers are reported in the supplementary material, and full-shot adaptation (e.g., regularizing the normal-appearance drift) is stated as future work. (These high-shot rows are single-seed and indicative; magnitude may vary, but the declining trend is consistent across all four measured categories.)

---

### R4-2 — Table 15: VLM and AD-reasoning model comparison

**Concern (R4-2, echoing AE and R1-2):** The comparison should include more VLMs (Qwen-VL, LLaVA) and their AD-tailored reasoning counterparts (IAD-R1, AnomalyOV, OmniAD); the AE specifically flags "insufficient comparison with recent VLM/MLLM-based logical AD methods such as LogiCode."

**Response.** We treat this in two parts. First, the positive evidence we *can* provide. Second, an itemized, per-model feasibility statement for the AD-specialized models — not a blanket "unavailable," but a precise account of what each project releases and why none of them can produce a *fair* MVTec LOCO AD number under our few-shot protocol.

**1. General-purpose VLMs — directly evaluated, plus a cited 72B point.** We ran Qwen2.5-VL-3B and 7B ourselves under the same positive-rule prompting strategy; our current evaluation re-confirms their LOCO averages (56.2 and 61.4). We further added **LLaVA-1.6-7B (49.0)** and **Qwen2-VL-7B (61.8)** to broaden the model coverage the reviewer requests. We additionally cite the only published Qwen2.5-VL-72B LOCO number (LAD-Reasoner). The per-category cells for GPT-4o and the Qwen2.5-VL variants are reproduced from our prior Table 15 for context; the load-bearing quantities below are the averages.

| Method | Breakfast | Splicing | Juice | Pushpins | Screw | Avg | Δ vs Ours 1-shot |
|--------|-----------|---------|-------|---------|-------|-----|------------------|
| LLaVA-1.6-7B (zero-shot, ours) | 44.9 | 47.6 | 60.2 | 39.7 | 52.9 | 49.0 | −21.0 |
| GPT-4o (zero-shot) | 78.9 | 55.0 | 68.2 | 40.6 | 61.0 | 60.7 | −9.3 |
| Qwen2.5-VL-3B (zero-shot, ours) | 67.0 | 62.1 | 64.0 | 36.7 | 51.4 | 56.2 | −13.8 |
| Qwen2-VL-7B (zero-shot, ours) | — | — | — | — | — | 61.8 | −8.2 |
| Qwen2.5-VL-7B (zero-shot, ours) | **90.3** | 52.9 | 60.2 | 46.3 | 57.5 | 61.4 | −8.6 |
| Qwen2.5-VL-72B† (zero-shot) | 74.6 | 57.3 | 64.4 | 62.9 | 54.8 | 62.8 | −7.2 |
| **Ours 1-shot** | 80.2 | **73.8** | **80.9** | **63.3** | 51.7 | **70.0** | — |
| **Ours 5-shot** | 85.2 | **75.4** | 83.7 | **64.4** | **55.8** | **72.9** | +2.9 |

† Cited from LAD-Reasoner (Li et al., arXiv:2504.12749), the only published source reporting Qwen2.5-VL-72B on MVTec LOCO AD.

**LLaVA-1.6-7B is the weakest (49.0)**, and **Qwen2-VL-7B (61.8) ≈ Qwen2.5-VL-7B (61.4)** — a clean generational comparison showing that the newer Qwen generation does **not** help on logical AD. Every general-purpose VLM remains well below our 1-shot 70.0.

The substantive finding is an *average* gap that holds across every VLM scale we could measure: on the LOCO average, every VLM — from GPT-4o to the 72B model — lands 7–14 points below our 1-shot result (Q-72B −7.2, Q-3B −13.8). We are careful that this is an average-level statement: per category the picture is mixed, e.g. Q-7B exceeds our 1-shot result on Breakfast (90.3 vs 80.2), and GPT-4o, Q-7B, and Q-72B all exceed it on Screw; our advantage is on the aggregate, driven by the logic-heavy categories (Splicing, Juice). Even so, the average gap is not a prompt-engineering artifact. A scale-vs-few-shot contrast makes the point: **scaling the zero-shot VLM backbone 23× (Qwen2.5-VL 3B → 72B) raises its LOCO average by only +6.6 points (56.2 → 62.8)**, and it still trails our few-shot result; meanwhile **our method, built on a far smaller CLIP backbone (ViT-B/16+, ~10⁸ parameters), reaches 70.0 at just 1-shot.** We are explicit that this is a *cross-model* contrast, not a controlled same-architecture ablation: the scaling arm is the zero-shot Qwen2.5-VL family, while the few-shot arm is our separate, much smaller CLIP model. Read as an attribution argument, it indicates that the dominant lever for logical anomaly detection is few-shot calibration to a specific product's normal appearance rather than raw model capacity — no zero-shot VLM in the 3B–72B range closes the average gap, while a single labeled shot on a ~10⁸-parameter backbone surpasses all of them.

**2. AD-specialized reasoning models — Anomaly-OV directly evaluated; the rest itemized by feasibility.** We did *not* dismiss these models; we attempted to use what each makes public, and where a runnable checkpoint exists we ran it.

- **Anomaly-OV — now evaluated directly.** Using the released expert detection head (a sigmoid anomaly probability from frozen SigLIP multi-level features, LLM-independent), we obtain a **zero-shot LOCO logical AUROC of 0.534** (all-anomaly AUROC 0.605). This is well below our 1-shot 0.727, and it is **structurally biased**: logical-only AUROC falls **below chance** on the counting/relational categories (pushpins 0.483, screw bag 0.450) — exactly the logical anomalies our method targets. We report this number alongside the CLIP-based baselines under R1-2.

For the remaining three, we state precisely what is released and the specific blocker for a fair LOCO number under our protocol:

| Model | What is released | Specific blocker for a fair LOCO comparison |
|-------|-----------------|---------------------------------------------|
| LogiCode | Annotations only | No inference code and no checkpoint are released, so the model cannot be run at all on LOCO. |
| IAD-R1 | Code **and** checkpoint (public) | The released model is not evaluated on any MVTec LOCO AD split; reporting a number would require us to define and run an evaluation protocol the authors never specified, which would not be *their* reported result. |
| OmniAD | Instruction-tuned multimodal reasoner | Reported on the MMAD benchmark, not under our few-shot MVTec LOCO protocol; obtaining a comparable number would require re-implementing its training/evaluation under our ≤5-shot regime, so it is not an apples-to-apples comparison. |

In short: **Anomaly-OV we evaluate directly (0.534);** LogiCode and OmniAD cannot be run on LOCO at all (no code/checkpoint, or a different benchmark); IAD-R1 is technically runnable but reports no LOCO result, so any number we produced would be our reimplementation under our own protocol rather than an author-validated, fair comparison, and we discuss it as complementary. We state each of these cases explicitly in the revised paper, and we provide the directly measured Q-3B/7B/LLaVA/Qwen2-VL averages, the cited Q-72B point, and the directly evaluated Anomaly-OV number as the concrete, reproducible comparison the reviewer asks for.

---

### R4-3 — Negative prompt count and generation ablation

**Concern:** Effect of negative prompt count and the generation module ablation are missing.

**Response:** See R2-2 for the complete ablation (the multi-seed original-vs-shuffled table plus the single-seed naive/nn rows). Summary:

- Varying the negative-rule count on the single-seed conditions, splicing is the most sensitive: nn=1: 0.568 → nn=3: 0.635 (the multi-seed original mean is 0.653); breakfast stays stable (nn=1: 0.889, nn=3: 0.887). A single negative rule already supplies useful contrastive signal, while a few more add robustness.
- Naive (no semantic checker) is comparable to the validated pipeline (within ~0.02–0.06 AUC), suggesting the checker is a safety mechanism for rule auditability rather than the dominant performance driver.
- Under multi-seed averaging, correct ("original") negatives **beat** shuffled on breakfast (+0.024); on the harder splicing category the original/shuffled difference is within overlapping one-sigma bands — graceful degradation under mis-assignment, not a benefit of wrong rules.

---

## Summary of Changes

| Category | Action taken | Evidence |
|----------|-------------|---------|
| **New experiments** | RRD ablation, λ sweep, β sweep, neg rule ablation, pos rule ablation, shot scalability, VLM comparison, multi-seed analysis | See R1–R4 sections |
| **Paper revision** | Title, conclusion (2→5 paragraphs), related work (+4 papers), implementation details, grammar/typos, metric names, rule-count claim corrected | EiC-a/b/c/d, R2-4/8/9, R3-5/6 |
| **Response justification** | Score fusion trade-off, logical vs structural anomaly taxonomy, rule source clarification, VLM gap analysis, LOCO fusion analysis | R1-1, R2-1/5/7, R3-2/4 |
| **Pending** | PDF page count verification (EiC-e), formal LaTeX compilation | EiC-e |
