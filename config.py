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

# CLIP model
# ViT-B-16 produces 512-d L2-normalised vectors for both images and text,
# enabling cosine similarity via inner product (IndexFlatIP).
CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 512

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
