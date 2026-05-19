"""
Attention visualisation for CLIP's text encoder using BertViz.

When a user types a text query, this module shows *which tokens each attention
head focuses on* across all 12 layers of the CLIP text encoder.  This answers
the question "why did the model retrieve these results?" at a mechanistic level:
later layers tend to show task-specific patterns where content words (landmark
names, architectural terms) attract disproportionate attention weight.

Implementation notes:
  - Uses the HuggingFace transformers CLIPTextModel to extract attention weights.
    This is a *different* model object from the open_clip model used for retrieval
    — same weights, different API - because open_clip does not expose per-layer
    attention tensors.
  - BertViz's head_view returns an HTML+JS+SVG blob that is rendered in Streamlit
    via st.components.v1.html.
  - A white background wrapper is applied so BertViz's dark SVG lines are visible
    on Streamlit's default dark theme.
"""

import sys
from pathlib import Path

import torch
from bertviz import head_view
from transformers import CLIPTextModel, CLIPTokenizerFast

sys.path.insert(0, str(Path(__file__).parents[2]))

# HuggingFace model ID — same weights as open_clip's "ViT-B-16"/"openai"
HF_MODEL = "openai/clip-vit-base-patch16"
_model = None
_tokenizer = None


def _load(device: str = "cpu"):
    """Lazy-load the HuggingFace CLIP text model.  Cached in module-level globals."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = CLIPTokenizerFast.from_pretrained(HF_MODEL)
        _model = CLIPTextModel.from_pretrained(HF_MODEL, output_attentions=True)
        _model.eval()
        _model.to(device)
    return _model, _tokenizer


def get_attention_html(query: str, device: str = "cpu") -> str:
    """
    Return a BertViz head_view HTML string for the given query.

    The HTML contains an interactive visualisation: 12 rows (one per transformer
    layer) × 8 columns (one per attention head).  Clicking a coloured square
    highlights that head's attention pattern over the query tokens.
    """
    model, tokenizer = _load(device)

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=77,  # CLIP hard token limit
    ).to(device)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len) tensors, one per layer
    html_obj = head_view(
        outputs.attentions,
        tokens,
        html_action="return",
    )

    # Wrap in white background so the dark SVG attention lines are visible
    # on Streamlit's dark theme
    html = f"""
    <div style="background:white; padding:12px; border-radius:8px;">
        {html_obj.data}
    </div>
    """
    return html
