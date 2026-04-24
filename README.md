# 🏛️ Where Is This? — Landmark Recognition & Storytelling System

A cross-modal system that identifies landmarks from photos and returns rich contextual descriptions, built with CLIP and FAISS.

## What It Does

- Upload a photo → get the top-3 matching landmarks with descriptions and confidence scores
- Type a text prompt → retrieve matching landmark images (bidirectional search)
- Upload multiple photos → get a generated travel narrative

## Tech Stack

- **CLIP ViT-B-16** — image and text embedding
- **Long-CLIP** — long-form text prompt support (Phase 3)
- **FAISS** — vector similarity search
- **Wikipedia & Wikidata APIs** — landmark metadata
- **Streamlit** — web interface

## Project Structure

src/data/          — data collection and downloading
src/embeddings/    — CLIP image and text encoders
src/retrieval/     — FAISS index and search logic
src/evaluation/    — metrics and bias audit
app/               — Streamlit UI
scripts/           — end-to-end pipeline runners

## Setup

```bash
uv sync
```

## Status

🚧 In development — NLP & Computer Vision, MSc Responsible AI Engineering
