"""
Text-to-image generation for the Where Is This? pipeline.

Given a text query (e.g. "ancient Roman arena where gladiators fought"),
generates a photorealistic image of the scene.  This completes the full
multimodal loop in the text search tab:

  text query → CLIP retrieval  (what real landmark matches?)
             → image generation (what does the model imagine?)

Two model presets are supported, selectable from the Streamlit sidebar:

  "sdxl"   stabilityai/stable-diffusion-xl-base-1.0
           Best quality. ~6 GB VRAM, fp16. ~10 s on L40S.
           Recommended - runs on the NVIDIA L40S server.

  "sd21"   stabilityai/stable-diffusion-2-1
           Lighter model. ~1.5 GB. Fallback for machines without a GPU.
"""

from __future__ import annotations

_NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, illustration, painting, drawing, "
    "watermark, text, deformed, ugly, bad anatomy"
)

_STYLE_SUFFIX = (
    ", photorealistic, architectural photography, golden hour lighting, "
    "sharp focus, highly detailed"
)

DEFAULT_STEPS = {"sdxl": 30, "sd21": 25}


def load_pipeline(model: str = "sdxl", device: str = "cuda"):
    """
    Load a diffusion pipeline for the given model preset.
    Called once per (model, device) pair; callers should cache the result.

    model:  "sdxl" | "sd21"
    device: "cuda" | "cpu"
    """
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32

    if model == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
        )
    elif model == "sd21":
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown model preset {model!r}. Choose 'sdxl' or 'sd21'.")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image(query: str, pipe, model: str = "sdxl", steps: int | None = None, guidance: float = 7.5):
    """
    Generate a PIL image from a text query.

    query:    the user's raw search text - used directly as the prompt
    pipe:     loaded pipeline from load_pipeline()
    model:    preset name - used to pick default steps
    steps:    denoising steps (defaults to DEFAULT_STEPS[model] if None)
    guidance: classifier-free guidance scale
    """
    if steps is None:
        steps = DEFAULT_STEPS.get(model, 25)

    result = pipe(
        prompt=query.strip() + _STYLE_SUFFIX,
        negative_prompt=_NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=1024,
        height=1024,
    )
    return result.images[0]  # PIL.Image
