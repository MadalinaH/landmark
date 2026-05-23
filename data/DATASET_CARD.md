# Dataset Card - Where Is This? Landmark Dataset

## Overview

A curated image-text dataset of 149 world landmarks, built to support
cross-modal retrieval using CLIP. Each landmark has a set of photographs
and a natural-language description sourced from Wikipedia.

---

## Composition

| Property | Value |
|----------|-------|
| Total landmarks | 149 |
| Landmarks with images | 148 (1 folder missing) |
| Total images | 1,184 |
| Images per landmark | 8 (uniform) |
| Description source | Wikipedia REST API + Wikidata SPARQL |
| Image source | Wikimedia Commons |
| Embedding dimension | 512-d (CLIP ViT-B-16) |

### Geographic distribution

| Region | Landmarks | Images | Share |
|--------|-----------|--------|-------|
| Europe | 39 | 312 | 26% |
| Asia | 30 | 240 | 20% |
| Americas | 25 | 200 | 17% |
| Vienna | 20 | 160 | 13% |
| Africa | 20 | 160 | 13% |
| Oceania | 10 | 80 | 7% |
| Natural | 5 | 40 | 3% |

Vienna is listed as a separate region because the dataset was built as
part of a university project at a Viennese institution, and Viennese
landmarks were collected with higher intentionality. Combined with the
broader Europe category, European landmarks account for 39% of the dataset.

---

## Collection Process

**Landmark selection:** Landmarks were selected manually to cover a broad
geographic spread across major world regions. Selection criteria were
(1) prominence in tourism and cultural heritage, (2) availability of
Wikimedia Commons images under a permissive licence, and (3) a Wikipedia
article with a substantive English description.

**Descriptions:** Fetched via the Wikipedia REST API (`/page/summary`) and
enriched with coordinates from the Wikidata SPARQL endpoint
(`wikibase:around` geospatial query). Descriptions are the Wikipedia
summary paragraph - typically 2–5 sentences covering the landmark's
location, historical significance, and architectural style.

**Images:** Downloaded from Wikimedia Commons. Exactly 8 images were
collected per landmark where possible, selected from the first page of
results for the landmark name query. Images are stored in one folder per
landmark under `data/images/`.

**Preprocessing:** Images are not preprocessed at storage time. The CLIP
preprocessing transform (resize to 224×224, normalise with ImageNet
mean/std) is applied at embedding time.

---

## Known Biases

**Geographic skew:** Europe + Vienna represent 39% of landmarks while
Oceania (7%) and Natural landmarks (3%) are substantially underrepresented.
This directly impacts retrieval accuracy: Oceanian and natural landmarks
have fewer semantically distinct neighbours in the embedding space, making
misclassification more likely. The geographic bias audit (see
`evaluation/bias_chart.png`) quantifies this per region.

**Language:** All descriptions are English-only. Landmark names in
non-Latin scripts (e.g. Fushimi Inari, Angkor Wat) are transliterated.
Queries in other languages are partially supported via CLIP's multilingual
pretraining but have not been evaluated.

**Image uniformity:** All landmarks have exactly 8 images regardless of
how visually diverse those images are. Landmarks with highly similar
training images (e.g. Museum of Natural History Vienna and Kunsthistorisches
Museum Vienna — architecturally identical twin buildings) are harder to
distinguish regardless of model capacity.

**Wikipedia coverage bias:** Landmark selection was constrained to sites
with substantive English Wikipedia articles. This systematically
underrepresents landmarks in regions with lower English-language Wikipedia
coverage, particularly rural Africa and parts of Asia and Oceania.

**Temporal snapshot:** Descriptions and images were collected at a single
point in time. Descriptions may become outdated; images reflect the site's
appearance at collection time.

---

## Sensitive Content

15 landmarks (10%) are flagged with `sensitive: true` and a
`sensitivity_reason` field in `landmarks.json`. Sensitivity categories:

| Category | Count | Examples |
|----------|-------|---------|
| Active religious sites with access restrictions | 5 | Dome of the Rock, Sheikh Zayed Grand Mosque, Koutoubia Mosque |
| Sites with photography prohibitions | 4 | Valley of the Kings, Uluru (certain areas) |
| Solemn memorial sites | 3 | Hiroshima Peace Memorial, Robben Island, Lincoln Memorial |
| Sacred indigenous sites | 2 | Uluru, Bagan |
| Political sensitivity | 1 | Potala Palace Tibet |

These flags are surfaced in the UI as a yellow warning box beneath the
result card. The flagging list was compiled manually and is not exhaustive —
new additions require manual review.

**By region:** Asia (7), Africa (4), Europe (2), Americas (1), Oceania (1).

---

## Intended Use

- Cross-modal landmark retrieval (image query → landmark, text query → landmark)
- CLIP fine-tuning research on a small domain-specific dataset
- Responsible AI education: demonstrating geographic bias, confidence
  calibration, sensitive site handling, and EXIF privacy warnings

---

## Out-of-Scope Use

- Face recognition or identification of individuals
- Surveillance or tracking
- Any use requiring precise legal or safety-critical identification of sites
  (the system is a research prototype; confidence scores are empirically
  calibrated but not formally validated)

---

## Licence

Images sourced from Wikimedia Commons are subject to their individual
licences (typically CC BY-SA or CC BY). Descriptions are derived from
Wikipedia content licenced under CC BY-SA 4.0. This dataset is intended
for non-commercial academic research only.
