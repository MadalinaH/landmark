"""
Central configuration for the Where Is This? landmark recognition system.

All constants that are referenced by more than one module live here so that
changing a value (e.g. confidence threshold, model name) only requires editing
this file.  The .env file in the project root is loaded automatically so that
secrets (ANTHROPIC_API_KEY) never appear in source code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root before reading any env vars
load_dotenv(Path(__file__).parent / ".env")

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# Paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"  # one sub-folder per landmark, 8 images each
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = ROOT / "evaluation"

# Backbone selection
# Set BACKBONE=siglip to run the full pipeline with SigLIP ViT-B-16 (Google/webli).
# Defaults to "clip" (OpenAI ViT-B-16) so existing indexes and results are untouched.
BACKBONE: str = os.getenv("BACKBONE", "clip").lower()

if BACKBONE == "siglip":
    # SigLIP ViT-B-16 - sigmoid loss, trained on WebLI (larger + multilingual).
    # Outputs 768-d embeddings (NOT 512-d like CLIP ViT-B-16).
    CLIP_MODEL = "ViT-B-16-SigLIP"
    CLIP_PRETRAINED = "webli"
    FAISS_INDEX_PATH = DATA_DIR / "faiss_index_siglip.bin"
    FAISS_TEXT_INDEX_PATH = DATA_DIR / "faiss_text_index_siglip.bin"
    _DEFAULT_CHECKPOINT = "siglip_finetuned_best.pt"
    EMBEDDING_DIM = 768
else:
    # CLIP ViT-B-16 - original OpenAI weights; default.
    CLIP_MODEL = "ViT-B-16"
    CLIP_PRETRAINED = "openai"
    FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
    FAISS_TEXT_INDEX_PATH = DATA_DIR / "faiss_text_index.bin"
    _DEFAULT_CHECKPOINT = "clip_finetuned_best.pt"
    EMBEDDING_DIM = 512

# Fine-tuned weights - set CLIP_WEIGHTS_PATH to a checkpoint path to use fine-tuned
# weights instead of the pretrained defaults.  None means use pretrained only.
# Example: CLIP_WEIGHTS_PATH=data/checkpoints/clip_finetuned_best.pt python3 ...
_clip_weights_env = os.getenv("CLIP_WEIGHTS_PATH")
CLIP_WEIGHTS_PATH = Path(_clip_weights_env) if _clip_weights_env else None

# Retrieval
TOP_K = 3

# Confidence thresholds calibrated empirically on the 5-image golden set:
#   image-to-image cosine scores: correct matches cluster at 0.85–0.94
#   cross-modal (text query) cosine scores: correct matches cluster at 0.28–0.35
# Scores below the threshold trigger a "low confidence" warning in the UI.
CONFIDENCE_THRESHOLD_IMAGE = 0.82
CONFIDENCE_THRESHOLD_TEXT = 0.25
CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD_TEXT  # default kept for backwards compat

# Data collection
NUM_IMAGES_PER_LANDMARK = 8
