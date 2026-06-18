# Revision baselines on MVTec LOCO AD (image-level AUROC ×100)

Reference: our method LOCO avg AUROC = **72.7 (1-shot) / 77.0 (5-shot)**.
Already in paper: AnomalyCLIP 61.9, AdaCLIP 59.9 (zero-shot, cross-dataset checkpoints).

## Reasoning-MLLM (AE headline / R4-2)

### Anomaly-OV (Anomaly-OneVision, CVPR'25) — zero-shot, expert detection head
Source: honda-research-institute/Anomaly-OneVision, released expert head + 0.5B base.
Score = sigmoid anomaly prob from frozen SigLIP multi-level features + released expert head (LLM-independent; 0.5B vs 7B does not affect this score).

| Category | All-anom AUROC | Logical-only AUROC |
|---|---|---|
| breakfast_box | 62.7 | 60.2 |
| juice_bottle | 73.6 | 66.8 |
| pushpins | 59.9 | 48.3 |
| screw_bag | 54.8 | 45.0 |
| splicing_connectors | 51.7 | 46.9 |
| **Average** | **60.5** | **53.4** |

Honest framing: even the strongest reasoning-MLLM's detection head, zero-shot, averages only 60.5 (all) / 53.4 (logical) AUROC on LOCO — below our 1-shot 72.7. It is structurally biased: logical-only AUROC falls **below chance** on the counting/relational categories (pushpins 48.3, screw_bag 45.0), the very logical anomalies our method targets. Caveat: zero-shot (not few-shot), and the score is a SigLIP patch-similarity detector, not LLM reasoning.

## Recent CLIP-based AD (R1-2)

> **Comparable metric = logical-only AUROC** (our paper's `evaluate()` logs `la_auc` = good vs logical_anomalies; structural is commented out). All-anomalies is secondary context only.

### MuSc (ICLR'24) — training-free, transductive zero-shot
Backbone ViT-L-14-336 (2× larger than our ViT-B-32), 336px (518 OOM'd under GPU sharing → mild lower bound). Scores each test image by mutual similarity over the *entire unlabeled test set* (transductive; uses no train images, no text). Not few-shot / not per-image.

| Category | All-anom | **Logical-only** |
|---|---|---|
| breakfast_box | 81.8 | 81.4 |
| juice_bottle | 90.1 | 86.7 |
| pushpins | 66.9 | 57.9 |
| screw_bag | 68.3 | 55.9 |
| splicing_connectors | 70.8 | 68.5 |
| **Average** | **75.6** | **70.1** |

Framing: MuSc is a strong recent zero-shot method, but **transductive (needs the whole test set jointly) and uses a 2× larger backbone**, yet on the comparable logical metric (70.1) it is still below our **few-shot per-image** 1-shot 72.7 / 5-shot 77.0 — and collapses on counting (pushpins 57.9, screw_bag 55.9), the same logical-anomaly weakness.

### APRIL-GAN (VAND, CVPR'23) — cross-dataset (MVTec-AD weights), ViT-L/14-336
3 seeds averaged. "log" = logical-only.

| Category | ZS log | k=1 log | k=5 log |
|---|---|---|---|
| breakfast_box | 62.3 | 57.6 | 58.3 |
| juice_bottle | 65.4 | 61.1 | 71.8 |
| pushpins | 57.7 | 47.4 | 47.4 |
| screw_bag | 45.9 | 49.8 | 49.6 |
| splicing_connectors | 65.1 | 56.6 | 59.5 |
| **Average** | **59.3** | **54.5** | **57.3** |

(all-anomalies avg: ZS 58.8 / k=1 62.9 / k=5 65.8.) Cross-dataset transfer with structural-defect prompts mismatched to LOCO logic; few-shot memory bank does not help logical anomalies. Same counting collapse (pushpins/screw_bag ~47-50).

### FINAL comparison (logical-only AUROC — the metric our paper reports)
| Method | Setting | Backbone | LOCO logical AUROC |
|---|---|---|---|
| Anomaly-OV (detector head) | zero-shot | SigLIP+LLaVA-OV | 53.4 |
| APRIL-GAN | few-shot k=5, cross-data | ViT-L/14-336 | 57.3 |
| AdaCLIP | zero-shot, cross-data | ViT-L | 59.9 |
| AnomalyCLIP | zero-shot, cross-data | ViT-L | 61.9 |
| MuSc | transductive zero-shot | ViT-L/14-336 | 70.1 |
| **Ours** | **few-shot 1/5-shot** | **ViT-B/32** | **72.7 / 77.0** |

**Takeaway**: across 5 recent CLIP/SigLIP/reasoning-MLLM detectors — several using a 2× larger ViT-L backbone and/or the easier transductive or zero-shot setting — none reaches our few-shot logical-AD numbers, and all collapse on the counting categories (pushpins/screw_bag), reinforcing that few-shot rule-grounded calibration, not backbone scale, is what closes the logical-anomaly gap.

---

## General-purpose VLM zero-shot accuracy (R4-2) — good vs logical, classification accuracy (%)
(Our method "accuracy" in the chatgpt_for_lad table = 70.0; paper already has Qwen2.5-VL-3B 56.2 / 7B 61.4 / 72B 62.8, GPT-4o 60.7.)

| Model | Setting | Avg acc | per-category (B/Sp/J/Pu/Sc) |
|---|---|---|---|
| LLaVA-1.6-Mistral-7B | zero-shot, single prompt | **49.0** | 44.9 / 47.6 / 60.2 / 39.7 / 52.9 |
| Qwen2-VL-7B | zero-shot, single prompt | **61.8** | (≈ Qwen2.5-VL-7B 61.4; clean generational comparison) |

Both below our 70.0. LLaVA is the weakest. Qwen2-VL ≈ Qwen2.5-VL (scaling generation doesn't help).

## Full-shot scalability (R4-1) — best_la_auc (our method), WinCLIP-fusion config
Single-seed (seed=0; multiple rows = few-shot sampling variance); high-shot n=1 (indicative). Killed before splicing/screw_bag-high finished — but the trend is consistent across all 4 measured categories.

| Category | shot50 | shot100 | shot200 | shot300 | WinCLIP-only branch |
|---|---|---|---|---|---|
| breakfast | 87.4 | 76.0 | 71.9 | 67.0 | ~66 (flat) |
| juice_bot | 84.4 | 76.2 | 72.1 | 71.3 | ~66 (flat) |
| pushpins | 67.9 | 60.9 | 58.3 | 58.2 | — |
| screw_bag | 57.4 | 54.1 | — | — | — |

**Finding (awkward for R4-1)**: our method's peak AUROC **declines monotonically** as shots grow 50→300, across all measured categories. Mechanism: early-stop is on test-AUROC; at high shots the model peaks at epoch 1–11 then overfits normal appearance, weakening the rule-grounded contrast. The WinCLIP-only branch stays flat (~66), so the decline is specific to our logic branch. **This is a real "few-shot specialist" characterization, NOT a scaling win.** Framing decision pending (A: report decline honestly / B: multi-seed re-run / C: reposition to ≤20-shot focus + supplementary). Confound: single-seed, high-shot n=1 → magnitude unreliable; a clean claim needs multi-seed.

## Rule-wording robustness, 5-category multi-seed (R1-5/R3-2) — best_la_auc, ours no-fusion 5-shot
New 3 categories, 4 seeds (0/7/42/123) each. (breakfast/splicing single-seed already in paper's table_positive_rule_robustness.)

| Category | original | vague | paraphrase | spread |
|---|---|---|---|---|
| juice_bottle | 80.1±1.2 | 78.5±1.4 | 79.7±2.0 | 1.6 |
| pushpins | 65.5±5.4 | 67.3±6.5 | 68.9±4.2 | 3.4 |
| screw_bag | 61.7±1.5 | 62.4±1.3 | 61.5±2.2 | 0.9 |

**Finding (clean, no controversy)**: across all 3 new categories, rewording rules as *vague* or *paraphrase* changes AUROC by at most 3.4 points (within across-seed std), and detection never collapses. Original is not always best (pushpins paraphrase 68.9 > original 65.5), consistent with our honest "wording is not critical; the framework tolerates substantial rewording" claim. Extends R1-5/R3-2 from 2 single-seed categories to **5 categories with multi-seed validation**.

## GPT-4o improved prompting (R2-7) — full test set (1136 imgs), no cap
| Prompt | Protocol | Avg acc | B / Sp / J / Pu / Sc |
|---|---|---|---|
| original (paper table) | per-rule querying (Algorithm 2) | 60.7 | 78.9/55.0/68.2/40.6/61.0 |
| simple (my script) | single combined prompt | 50.9 | 44.9/47.1/70.3/39.7/52.5 |
| **CoT (my script)** | single prompt, step-by-step | **67.3** | 84.3/53.7/72.0/68.6/57.9 |

⚠️ **Protocol caveat**: the paper's 60.7 used *per-rule* querying; my "simple" (50.9) uses a *single combined* prompt (weaker for GPT-4o, esp. breakfast 44.9 vs 78.9). So 50.9 is NOT the paper baseline. The defensible R2-7 statement: an improved **CoT** prompt raises GPT-4o to **67.3% (> the original 60.7%)**, still **below our 70.0%**; gains concentrate on reasoning-amenable categories (breakfast 84.3) while counting stays weak (screw 57.9, splicing 53.7) → better prompting narrows but does not close the gap; bottleneck is few-shot calibration, not prompt design. (For a perfectly same-protocol delta one could reproduce per-rule simple+CoT, but that is ~13k API calls.)
