# CLIP Fine-Tuning Evaluation Report

## Setup

| | Baseline | Fine-tuned |
|---|---|---|
| Model | CLIP ViT-B-16 (OpenAI pretrained) | CLIP ViT-B-16 + partial fine-tune |
| Trainable parameters | 0 (frozen) | 21,135,872 / 149,620,737 (14.1%) |
| Fine-tuned layers | - | Last 2 transformer blocks (image + text encoder) + projection heads |
| Training pairs | - | ~1,184 (150 landmarks × ~8 images, 1 missing folder) |
| Train / val split | - | 90% / 10% (1,066 train / 118 val) |
| Optimizer | - | AdamW, lr=5e-6, weight decay=0.01 |
| Schedule | - | Cosine annealing |
| Mixed precision | - | bf16 (NVIDIA L40S) |
| Batch size | - | 128 |
| Epochs | - | 20 (best checkpoint at epoch 11) |
| Loss (InfoNCE) | - | train 1.002→0.496 / val 1.041→0.877 |
| Hardware | - | NVIDIA L40S 46 GB |
| Training time | - | ~5 minutes |

---

## Results on 5-Image Golden Set (image-to-image mode)

| Query | Expected | Baseline Top-1 | Score | Fine-tuned Top-1 | Score | Δ Score |
|---|---|---|---|---|---|---|
| schoenbrunn_test.jpg | Schönbrunn Palace | Schönbrunn Palace ✓ | 0.836 | Schönbrunn Palace ✓ | 0.853 | **+0.017** |
| eiffel_test.jpg | Eiffel Tower | Eiffel Tower ✓ | 0.937 | Eiffel Tower ✓ | 0.946 | **+0.009** |
| colosseum_test.jpg | Colosseum | Trevi Fountain ✗ | 0.831 | Trevi Fountain ✗ | 0.834 | +0.003 |
| sagrada_test.jpg | Sagrada Família | Sagrada Familia ✓ | 0.928 | Sagrada Familia ✓ | 0.921 | -0.007 |
| stephansdom_test.jpg | Stephansdom | Stephansdom ✓ | 0.888 | Stephansdom ✓ | 0.902 | **+0.014** |

### Aggregate metrics

| Metric | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Hits@1 | 4/5 (80%) | 4/5 (80%) | 0 |
| Hits@3 | 4/5 (80%) | 4/5 (80%) | 0 |
| MRR | 0.800 | 0.800 | 0 |
| Mean cosine score (correct matches) | 0.900 | 0.906 | **+0.006** |

---

## Analysis

### Why Hits@1 and MRR are identical

Both models retrieve the nearest available Roman landmark (Trevi Fountain, score ~0.83) in its place.

### Fine-tuning effect on confidence scores

On the 4 landmarks present in the index, fine-tuning improved cosine similarity scores for 3 out of 4:

- **Schönbrunn +0.017**, **Stephansdom +0.014**, **Eiffel +0.009** - consistent improvement across European landmarks
- **Sagrada Família -0.007** - marginal regression, within noise given the small evaluation set

The mean cosine score on correct matches increased from 0.900 to 0.906. Higher similarity scores mean the model is more confident when it retrieves the correct landmark, which directly benefits the confidence calibration thresholds in the UI.

### Training dynamics

The validation loss plateaued between epochs 9–11 (best: 0.8776 at epoch 11) and did not improve meaningfully in epochs 12–20, indicating the model converged with the available data. This is expected: with ~1,200 training pairs, the contrastive loss has limited in-batch negatives per step (~9 batches of 128) and limited diversity.

---

## Limitations

- **Evaluation set is too small**: 5 images is insufficient to draw statistically significant conclusions. A 50-image golden set spanning multiple continents would allow meaningful before/after comparison.
- **Missing Colosseum data**: the only failing case cannot be attributed to model quality.
- **Score improvements may be noise**: a ±0.01 change in cosine similarity on 4 samples cannot be distinguished from random variation.

---

## Recommendations

| Option | Expected impact | Effort |
|---|---|---|
| Add Colosseum images to server and rebuild index | Restores 5/5 baseline, enables fair comparison | Low |
| Expand to 20–30 images per landmark | More in-batch negatives → stronger contrastive signal | Medium |
| Expand golden set to 50+ images across continents | Statistically meaningful evaluation | Medium |
| Geographic bias audit (Hits@1 per continent) | Directly addresses Responsible AI angle | Low–Medium |

---

*Evaluation run: 2026-05-19 - NVIDIA L40S, CUDA 13.0*
