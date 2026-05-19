"""
CLIP image encoder for the Where Is This? pipeline.

Responsibilities:
  - Load the CLIP ViT-B-16 model via open_clip
  - For each landmark folder under data/images/, average the CLIP embeddings
    of all images (up to NUM_IMAGES_PER_LANDMARK) into a single L2-normalised
    512-d vector - this is the landmark's "image fingerprint"
  - Save embeddings.npy (shape N×512) and metadata.json (index → landmark dict)
  - Embed a single query image at search time

Averaging multiple images per landmark makes the representation more robust
than any single photo and reduces sensitivity to camera angle and lighting.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CLIP_MODEL, CLIP_PRETRAINED, DATA_DIR, IMAGES_DIR
from src.utils import sanitize_folder_name


def _load_model(device: str = "cpu"):
    """Load CLIP model + preprocessing transform. Called once and cached by callers."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    model.to(device)
    return model, preprocess


def embed_landmark_folder(
    folder: Path, model, preprocess, device: str = "cpu"
) -> np.ndarray | None:
    """
    Return a single L2-normalised embedding for a landmark folder by averaging
    all valid image embeddings inside it.  Returns None if the folder is empty
    or all images fail to load.
    """
    image_paths = sorted(
        p
        for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        return None

    embeddings = []
    for img_path in image_paths:
        try:
            img = (
                preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                emb = model.encode_image(img)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalise per image
            embeddings.append(emb.cpu().numpy()[0])
        except Exception as e:
            warnings.warn(f"Skipping {img_path.name}: {e}")

    if not embeddings:
        return None

    # Average then re-normalise so the result remains a unit vector
    avg = np.mean(embeddings, axis=0).astype(np.float32)
    avg = avg / np.linalg.norm(avg)
    return avg


def build_image_embeddings(device: str = "cpu") -> tuple[np.ndarray, list[dict]]:
    """
    Iterate over all landmarks in landmarks.json, embed each folder, and
    return (embeddings array of shape N×512, metadata list of N landmark dicts).

    Landmarks whose image folder is missing or empty are skipped with a warning,
    so metadata.json may be a strict subset of landmarks.json - this is why the
    image FAISS index uses its own metadata.json rather than landmarks.json.
    """
    landmarks = json.loads((DATA_DIR / "landmarks.json").read_text())
    model, preprocess = _load_model(device)

    embeddings = []
    metadata = []

    for lm in tqdm(landmarks, desc="Embedding landmarks"):
        folder_name = sanitize_folder_name(lm["name"])
        folder = IMAGES_DIR / folder_name

        if not folder.exists():
            warnings.warn(
                f"Missing folder for '{lm['name']}' ({folder_name}), skipping."
            )
            continue

        emb = embed_landmark_folder(folder, model, preprocess, device)
        if emb is None:
            warnings.warn(f"No valid images for '{lm['name']}', skipping.")
            continue

        embeddings.append(emb)
        metadata.append(lm)

    if not embeddings:
        raise RuntimeError(
            "No embeddings produced - are images present in data/images/?"
        )

    return np.stack(embeddings, axis=0), metadata


def save_embeddings(embeddings: np.ndarray, metadata: list[dict]) -> None:
    out_dir = DATA_DIR
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(f"Saved {len(metadata)} embeddings → {out_dir / 'embeddings.npy'}")


def embed_single_image(
    image_path: Path, model=None, preprocess=None, device: str = "cpu"
) -> np.ndarray:
    """
    Embed a single query image at search time.
    Loads the model if not provided (slow path, used only in evaluation scripts).
    The returned vector is L2-normalised and ready for FAISS IndexFlatIP search.
    """
    if model is None or preprocess is None:
        model, preprocess = _load_model(device)
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)
