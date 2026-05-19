"""
CLIP text encoder for the Where Is This? pipeline.

Responsibilities:
  - Encode each landmark's Wikipedia description into a 512-d L2-normalised
    vector using CLIP's text encoder
  - Handle descriptions longer than CLIP's 77-token hard limit by splitting
    into sentences, grouping them into ≤77-token chunks, encoding each chunk
    separately, and averaging the resulting embeddings
  - Save text_embeddings.npy (shape N×512)

The resulting text index is the *primary* retrieval target: a query image is
encoded as an image embedding and compared against these text embeddings
(cross-modal retrieval).  This exploits CLIP's shared image-text embedding
space - images and their descriptions map to nearby vectors without any
fine-tuning.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import open_clip
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CLIP_MODEL, CLIP_PRETRAINED, CLIP_WEIGHTS_PATH, DATA_DIR

_MAX_TOKENS = 77  # CLIP hard limit — longer sequences are silently truncated


def _load_model(device: str = "cpu"):
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    if CLIP_WEIGHTS_PATH is not None:
        import torch
        checkpoint = torch.load(CLIP_WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded fine-tuned weights from {CLIP_WEIGHTS_PATH}")
    model.eval()
    model.to(device)
    return model, tokenizer


def _embed_text_chunks(text: str, model, tokenizer, device: str) -> np.ndarray:
    """
    Encode text that may exceed 77 tokens.

    Strategy: split on sentence boundaries, accumulate sentences into a chunk
    until adding the next sentence would exceed the token limit, then encode
    the chunk and start a new one.  Average all chunk embeddings and
    re-normalise to produce a single unit vector.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        tokens = tokenizer([candidate])[0]
        # Count non-padding tokens (non-zero values after the leading BOS token)
        n_tokens = int((tokens != 0).sum())
        if n_tokens > _MAX_TOKENS and current:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)

    embeddings = []
    for chunk in chunks:
        try:
            toks = tokenizer([chunk]).to(device)
            with torch.no_grad():
                emb = model.encode_text(toks)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy()[0])
        except Exception as e:
            warnings.warn(f"Failed to encode chunk: {e}")

    if not embeddings:
        return np.zeros(512, dtype=np.float32)

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    avg = avg / (
        np.linalg.norm(avg) + 1e-8
    )  # epsilon avoids divide-by-zero on zero vectors
    return avg


def build_text_embeddings(device: str = "cpu") -> tuple[np.ndarray, list[dict]]:
    """
    Embed all landmark descriptions from landmarks.json.
    Returns (embeddings array of shape N×512, metadata list of N landmark dicts).
    Unlike the image encoder, every landmark in landmarks.json is included
    because all have descriptions (fetched by scripts/fetch_descriptions.py).
    """
    landmarks = json.loads((DATA_DIR / "landmarks.json").read_text())
    model, tokenizer = _load_model(device)

    embeddings = []
    metadata = []

    for lm in tqdm(landmarks, desc="Embedding descriptions"):
        desc = lm.get("description", "")
        if not desc:
            warnings.warn(f"No description for '{lm['name']}', using name only.")
            desc = lm["name"]

        emb = _embed_text_chunks(desc, model, tokenizer, device)
        embeddings.append(emb)
        metadata.append(lm)

    return np.stack(embeddings, axis=0), metadata


def save_text_embeddings(embeddings: np.ndarray, metadata: list[dict]) -> None:
    out_dir = DATA_DIR
    np.save(out_dir / "text_embeddings.npy", embeddings)
    print(f"Saved {len(metadata)} text embeddings → {out_dir / 'text_embeddings.npy'}")
