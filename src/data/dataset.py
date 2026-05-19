"""
PyTorch Dataset for CLIP fine-tuning on the landmark image-text pairs.

Each landmark has one Wikipedia description and N images stored under
data/images/<landmark_folder>/. This Dataset expands those into individual
(image_tensor, text_tokens) pairs - one per image file - so the contrastive
loss can pull each image closer to its description in the shared embedding space.

With 150 landmarks × ~8 images, the full dataset is ~1200 pairs.  That is small
for contrastive learning, so the training script fine-tunes only the last two
transformer blocks and the projection heads rather than the full model.
"""

import json
import warnings
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import DATA_DIR, IMAGES_DIR, CLIP_MODEL
from src.utils import sanitize_folder_name

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class LandmarkDataset(Dataset):
    """
    Returns (image_tensor, text_tokens) pairs for all landmark images.

    image_tensor: float32 tensor of shape (3, H, W) after open_clip's
                  preprocessing transform — ready for model.encode_image().
    text_tokens:  int32 tensor of shape (77,) from open_clip's tokenizer
                  - ready for model.encode_text().
    """

    def __init__(self, preprocess, tokenizer, landmarks_json: Path = DATA_DIR / "landmarks.json"):
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._pairs: list[tuple[Path, str]] = []

        landmarks = json.loads(landmarks_json.read_text())
        missing = 0
        for lm in landmarks:
            folder = IMAGES_DIR / sanitize_folder_name(lm["name"])
            desc = lm.get("description", "") or lm["name"]
            if not folder.exists():
                missing += 1
                continue
            images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
            for img_path in images:
                self._pairs.append((img_path, desc))

        if missing:
            warnings.warn(f"{missing} landmark folders not found - skipped.")
        if not self._pairs:
            raise RuntimeError("No image-text pairs found. Are images in data/images/?")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, desc = self._pairs[idx]
        try:
            image = self._preprocess(Image.open(img_path).convert("RGB"))
        except Exception as e:
            warnings.warn(f"Failed to load {img_path}: {e} - using blank image.")
            image = torch.zeros(3, 224, 224)
        tokens = self._tokenizer([desc])[0]
        return image, tokens
