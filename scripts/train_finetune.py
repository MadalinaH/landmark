"""
Fine-tune CLIP ViT-B-16 on the landmark image-text pairs.

Strategy
--------
We have ~1200 image-text pairs (150 landmarks × ~8 images).  Training the full
model on this tiny set would overfit immediately.  Instead we freeze all layers
except the last two transformer blocks of both encoders and the final projection
heads - roughly 20% of parameters.  This is enough to pull landmark-specific
visual concepts closer to their Wikipedia descriptions without destroying the
general representations learned on 400 M image-text pairs.

Hardware assumptions
--------------------
NVIDIA L40S, 46 GB VRAM, CUDA 13.  bf16 mixed precision is enabled.
Batch size 128 fits comfortably; increase to 256 if VRAM allows.

Output
------
Checkpoint saved to data/checkpoints/clip_finetuned.pt every SAVE_EVERY epochs
and at the end of training.  Set CLIP_WEIGHTS_PATH in config.py (or .env) to
point at this file and the existing pipeline will use the fine-tuned weights.

Usage
-----
    python scripts/train_finetune.py [--epochs 20] [--batch-size 128] [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import BACKBONE, CLIP_MODEL, CLIP_PRETRAINED, DATA_DIR
from src.data.dataset import LandmarkDataset

SAVE_EVERY = 5  # save checkpoint every N epochs
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Checkpoint names are backbone-prefixed so CLIP and SigLIP runs never collide.
_CKPT_BEST = CHECKPOINT_DIR / f"{BACKBONE}_finetuned_best.pt"
_CKPT_FINAL = CHECKPOINT_DIR / f"{BACKBONE}_finetuned.pt"


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def _freeze_except_last_n_blocks(model, n: int = 2) -> None:
    """
    Freeze all parameters, then unfreeze the last n transformer blocks of
    both the image and text encoders, plus their projection heads.

    CLIP and SigLIP have different internal structures in open_clip:
      CLIP:    visual.transformer.resblocks / transformer.resblocks
      SigLIP:  visual.trunk.blocks (TimmModel) / text.transformer.resblocks
    """
    for param in model.parameters():
        param.requires_grad = False

    if BACKBONE == "siglip":
        # SigLIP visual encoder (TimmModel)
        for block in model.visual.trunk.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Attention-pool head serves as the image projection
        for param in model.visual.trunk.attn_pool.parameters():
            param.requires_grad = True

        # SigLIP text encoder
        for block in model.text.transformer.resblocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(model.text, "text_projection") and model.text.text_projection is not None:
            model.text.text_projection.requires_grad = True
    else:
        # CLIP visual encoder
        for block in model.visual.transformer.resblocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(model.visual, "proj") and model.visual.proj is not None:
            model.visual.proj.requires_grad = True

        # CLIP text encoder
        for block in model.transformer.resblocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(model, "text_projection") and model.text_projection is not None:
            model.text_projection.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ---------------------------------------------------------------------------
# Contrastive loss (InfoNCE / CLIP loss)
# ---------------------------------------------------------------------------

def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric cross-entropy loss over cosine similarities, scaled by logit_scale.
    Identical to the loss used in the original CLIP paper.
    """
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    labels = torch.arange(len(image_features), device=image_features.device)
    loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = args.device
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device)
    _freeze_except_last_n_blocks(model, n=args.unfreeze_blocks)
    model.train()

    # Dataset - 90% train / 10% validation split
    dataset = LandmarkDataset(preprocess, tokenizer)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Dataset: {train_size} train / {val_size} val pairs")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Optimizer + LR schedule
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # bf16 mixed precision scaler (bf16 doesn't need GradScaler, but fp16 does)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using dtype: {dtype}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for images, texts in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            images = images.to(device)
            texts = texts.to(device)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                loss = clip_loss(image_features, text_features, model.logit_scale.exp())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                images = images.to(device)
                texts = texts.to(device)
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(texts)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    loss = clip_loss(image_features, text_features, model.logit_scale.exp())
                val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={mean_train:.4f}  val={mean_val:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            _save(model, _CKPT_BEST)
            print(f"  ↳ new best val loss - saved {_CKPT_BEST.name}")

        if epoch % SAVE_EVERY == 0:
            _save(model, CHECKPOINT_DIR / f"{BACKBONE}_finetuned_epoch{epoch:03d}.pt")

    _save(model, _CKPT_FINAL)
    print(f"\nTraining complete. Final checkpoint: {_CKPT_FINAL}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"\nTo use in the pipeline:")
    print(f"  BACKBONE={BACKBONE} CLIP_WEIGHTS_PATH={_CKPT_BEST} python3 scripts/ablation.py")


def _save(model, path: Path) -> None:
    torch.save({"state_dict": model.state_dict()}, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on landmark data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="Number of last transformer blocks to fine-tune")
    args = parser.parse_args()
    print(f"Device: {args.device}  |  Epochs: {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    train(args)
