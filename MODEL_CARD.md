# Model Card - Where Is This? CLIP ViT-B-16

## Model description

A fine-tuned CLIP ViT-B-16 model adapted for cross-modal landmark retrieval.
Given a photo or a natural-language description, the model encodes the input
into a 512-dimensional vector and retrieves the closest landmark from a
pre-built FAISS index using cosine similarity.

The base model is OpenAI's CLIP ViT-B-16, pretrained on 400 million
image-text pairs. Fine-tuning adapts the final layers to the landmark domain
without losing the general visual and linguistic representations learned
during pretraining.

---

## Intended use

**Primary use:** Landmark identification from photos and natural-language
descriptions, for educational and research purposes.

**Intended users:** Researchers, students, and developers exploring
cross-modal retrieval, responsible AI, and vision-language models.

**Out-of-scope use:**
- Face recognition or identification of individuals
- Surveillance or tracking of people or vehicles
- Safety-critical or legally binding identification of sites
- Any commercial deployment without further validation

---

## Training

### Base model
- Architecture: CLIP ViT-B-16
- Pretrained weights: OpenAI (`openai` preset via `open_clip`)
- Pretraining data: 400M image-text pairs (WIT dataset)

### Fine-tuning dataset
- 1,184 image-text pairs across 148 landmarks
- Images: Wikimedia Commons (8 per landmark)
- Texts: Wikipedia summary descriptions
- See `data/DATASET_CARD.md` for full dataset documentation

### Fine-tuning procedure

| Hyperparameter | Value |
|----------------|-------|
| Loss | Symmetric InfoNCE (contrastive) |
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Weight decay | 0.01 |
| LR schedule | Cosine annealing |
| Batch size | 128 |
| Epochs | 20 |
| Precision | bf16 |
| Hardware | NVIDIA L40S |
| Training time | ~5 minutes |

### Frozen layers
86% of parameters are frozen. Only the following are fine-tuned:
- Last 2 transformer blocks of the image encoder (`visual.transformer.resblocks[-2:]`)
- Last 2 transformer blocks of the text encoder (`transformer.resblocks[-2:]`)
- Image projection head (`visual.proj`)
- Text projection head (`text_projection`)

This strategy preserves general representations while adapting the final
projections to the landmark domain, reducing overfitting on the small dataset.

### Checkpoint selection
Best checkpoint selected by validation loss on a 10% hold-out split
(`clip_finetuned_best.pt`). Final epoch checkpoint also saved
(`clip_finetuned.pt`).

---

## Evaluation

Full evaluation methodology and results are documented in
`evaluation/FINDINGS.md`. Summary:

### Image retrieval (leave-one-out, 148 landmarks)

| Condition | Hits@1 | Hits@3 | MRR | 95% CI |
|-----------|--------|--------|-----|--------|
| Baseline CLIP | 141/148 (95%) | 147/148 (99%) | 0.973 | [0.91, 0.99] |
| Fine-tuned CLIP | 144/148 (97%) | 147/148 (99%) | 0.983 | [0.95, 0.99] |

Fine-tuning improves Hits@1 by +2% and raises the CI lower bound from
0.91 to 0.95, indicating more consistent performance across landmarks.

### Text retrieval (149 landmark name queries)

Both baseline and fine-tuned models achieve 99% Hits@1 on landmark name
queries, confirming that CLIP's shared embedding space generalises well
to the landmark domain for text queries even without fine-tuning the text
encoder specifically.

### Known failure modes

| Failure | Cause |
|---------|-------|
| Museum of Natural History ↔ Kunsthistorisches Museum | Architecturally identical twin buildings - visually indistinguishable |
| Fushimi Inari ↔ Senso-ji | Both feature red Japanese religious architecture |
| Louvre → Abu Simbel | Query image angle shares visual features with index embedding |

---

## Limitations

**Geographic bias:** The training dataset skews heavily towards European
landmarks (39% Europe + Vienna). Counterintuitively, the per-region audit
(`evaluation/bias_chart.png`) shows the *most*-represented regions, Vienna
(90%) and Europe (97%), score lowest on Hits@1, while underrepresented
regions (Oceania, Africa, Natural) score 100%. Representation count alone
does not predict accuracy - visual homogeneity within a region (e.g.
Vienna's many similar Baroque/Gothic churches) appears to matter more than
how many examples exist.

**Small dataset:** 1,184 training pairs is small for fine-tuning a
vision-language model. Partial freezing mitigates overfitting but does
not eliminate it.

**Fixed vocabulary:** The model retrieves from a closed set of 149
landmarks. It cannot identify landmarks outside this set and does not
indicate when a query landmark is absent.

**Language:** Text queries are English-only in evaluation. Multilingual
queries are partially supported via CLIP's pretraining but have not been
formally evaluated.

**Confidence scores are not probabilities:** Cosine similarity scores are
used as a proxy for confidence. The threshold (0.82 for image search) is
calibrated empirically on a 5-image golden set and should not be treated
as a formal probability of correctness.

---

## Responsible AI considerations

**Confidence calibration:** Scores below the empirical threshold surface a
⚠️ Low confidence warning in the UI rather than presenting results as
certain.

**Sensitive sites:** 15 landmarks in the index are flagged with access and
photography restrictions. Results for these landmarks show a contextual
warning in the UI.

**Geographic bias transparency:** A geographic bias audit (Hits@1 per
region) is run automatically and displayed in the Model Report tab of the
application, making performance disparities visible to users.

**Grounded generation:** The Instagram post generation feature uses
retrieved Wikipedia descriptions as the factual basis for Claude-generated
text. Claude is explicitly instructed not to invent facts beyond what the
descriptions contain.

---

## How to use

```python
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="openai"
)

# Load fine-tuned weights
checkpoint = torch.load("data/checkpoints/clip_finetuned_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Encode an image
from PIL import Image
img = preprocess(Image.open("query.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    emb = model.encode_image(img)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # L2-normalise
```

See `src/embeddings/image_encoder.py` and `src/embeddings/text_encoder.py`
for the full inference pipeline used in the application.

---

## Citation

This model was developed as part of an MSc project in Responsible AI
Engineering. If you use it, please cite the project repository.
