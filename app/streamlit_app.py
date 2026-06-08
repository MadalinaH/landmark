import os
import sys
import tempfile
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parents[1]))

from config import (
    ANTHROPIC_API_KEY,
    CLIP_WEIGHTS_PATH,
    CONFIDENCE_THRESHOLD_IMAGE,
    CONFIDENCE_THRESHOLD_TEXT,
    DATA_DIR,
    EVAL_DIR,
    IMAGES_DIR,
    TOP_K,
)
from src.explainability.attention import get_attention_html
from src.generation.image_gen import generate_image, load_pipeline
from src.generation.travel_story import generate_story
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.search import LandmarkSearcher
from src.retrieval.text_search import TextSearcher
from src.utils import extract_gps, sanitize_folder_name

st.set_page_config(page_title="Where Is This?", page_icon="🌍", layout="wide")

# Global CSS
st.markdown(
    """
<style>
/* Hero */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.07);
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, #e0e7ff, #a5b4fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.hero-stat {
    text-align: center;
}
.hero-stat .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a5b4fc;
    line-height: 1;
}
.hero-stat .label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* Result cards */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.result-card:hover {
    border-color: rgba(165,180,252,0.35);
    background: rgba(165,180,252,0.04);
}
.result-card .card-header {
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
}
.result-card .card-thumb {
    width: 80px;
    height: 80px;
    border-radius: 10px;
    object-fit: cover;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.08);
}
.result-card .card-thumb-placeholder {
    width: 80px;
    height: 80px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
}
.result-card .card-info {
    flex: 1;
    min-width: 0;
}
.result-card .card-rank {
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
}
.result-card .card-name {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.25rem 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.result-card .card-region {
    font-size: 0.8rem;
    color: #64748b;
}
.result-card .card-score {
    text-align: right;
    flex-shrink: 0;
}
.score-badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1;
}
.score-high {
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.3);
}
.score-low {
    background: rgba(245,158,11,0.15);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
}
.score-label {
    font-size: 0.68rem;
    margin-top: 0.25rem;
    text-align: right;
}
.score-label.high { color: #34d399; }
.score-label.low  { color: #fbbf24; }
.result-card .card-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.55;
    margin-top: 0.85rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.sensitive-warning {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    background: rgba(234,179,8,0.08);
    border: 1px solid rgba(234,179,8,0.3);
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    margin-top: 0.75rem;
    font-size: 0.82rem;
    color: #fde047;
    line-height: 1.45;
}

/* Section headings */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
    margin-top: 0.25rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #94a3b8;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 0.5rem 1.1rem !important;
    border-radius: 8px !important;
    transition: color 0.15s ease, background 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #94a3b8 !important;
    background: rgba(255,255,255,0.04) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #fff !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    font-weight: 600 !important;
}
/* Hide the default underline indicator */
button[data-baseweb="tab"][aria-selected="true"] div,
button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] {
    background: transparent !important;
}

/* Privacy warning */
.gps-warn {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    font-size: 0.82rem;
    color: #fbbf24;
    line-height: 1.45;
}

/* Story output */
.story-box {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    font-size: 1rem;
    line-height: 1.8;
    color: #cbd5e1;
}
.story-disclaimer {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 1rem;
    font-style: italic;
}

/* Identified landmark chips */
.lm-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    color: #cbd5e1;
    margin: 0.25rem 0.25rem 0.25rem 0;
}
.lm-chip.warn {
    border-color: rgba(245,158,11,0.35);
    color: #fbbf24;
}
</style>
""",
    unsafe_allow_html=True,
)


# Cached resources
@st.cache_resource(show_spinner="Loading model and index…")
def get_searcher() -> LandmarkSearcher:
    return LandmarkSearcher(mode="image")


@st.cache_resource(show_spinner="Loading text search…")
def get_text_searcher() -> TextSearcher:
    return TextSearcher()


@st.cache_resource(show_spinner="Loading hybrid search…")
def get_hybrid_searcher() -> HybridSearcher:
    return HybridSearcher()


@st.cache_resource(show_spinner="Loading Stable Diffusion…")
def get_image_pipeline(model: str = "turbo", device: str = "cuda"):
    return load_pipeline(model=model, device=device)


def get_thumbnail(landmark_name: str) -> Path | None:
    folder = IMAGES_DIR / sanitize_folder_name(landmark_name)
    if not folder.exists():
        return None
    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    return images[0] if images else None


def render_result(result, rank: int) -> None:
    """Render a single search result as a styled card."""
    thumb = get_thumbnail(result.name)
    is_low = result.low_confidence

    score_class = "score-low" if is_low else "score-high"
    label_class = "low" if is_low else "high"
    label_text = "⚠ Low confidence" if is_low else "✓ High confidence"

    if thumb:
        import base64

        thumb_b64 = base64.b64encode(thumb.read_bytes()).decode()
        ext = thumb.suffix.lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        thumb_html = (
            f'<img class="card-thumb" src="data:image/{mime};base64,{thumb_b64}" />'
        )
    else:
        thumb_html = '<div class="card-thumb-placeholder">🏛</div>'

    region_html = (
        f'<div class="card-region">📍 {result.region}</div>' if result.region else ""
    )
    sensitive_html = (
        f'<div class="sensitive-warning">⚠️ <span>{result.sensitivity_reason}</span></div>'
        if getattr(result, "sensitive", False) and result.sensitivity_reason
        else ""
    )

    st.markdown(
        f"""
    <div class="result-card">
      <div class="card-header">
        {thumb_html}
        <div class="card-info">
          <div class="card-rank">Match #{rank}</div>
          <div class="card-name">{result.name}</div>
          {region_html}
        </div>
        <div class="card-score">
          <span class="score-badge {score_class}">{result.score:.1%}</span>
          <div class="score-label {label_class}">{label_text}</div>
        </div>
      </div>
      <div class="card-desc">{result.description}</div>
      {sensitive_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    if result.lat is not None and result.lon is not None:
        df = pd.DataFrame({"lat": [result.lat], "lon": [result.lon]})
        st.map(df, zoom=10)


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    top_k = st.slider("Results to show", min_value=1, max_value=5, value=TOP_K, step=1)

    st.markdown("**Image search threshold**")
    img_conf_threshold = st.slider(
        "Image confidence",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD_IMAGE,
        step=0.01,
        format="%.2f",
        help="Correct matches score ~0.85-0.94 on the golden set",
    )

    st.markdown("**Text search threshold**")
    txt_conf_threshold = st.slider(
        "Text confidence",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD_TEXT,
        step=0.01,
        format="%.2f",
        help="Cross-modal text scores: correct matches ~0.28–0.35",
    )

    st.divider()
    st.markdown("**Image generation**")
    enable_image_gen = st.toggle(
        "Generate image from query",
        value=False,
        help="Generates an image from your text query using Stable Diffusion.",
    )
    if enable_image_gen:
        sd_model = st.selectbox(
            "Model",
            ["sdxl", "sd21"],
            index=0,
            help="sdxl: best quality (GPU, ~10s) · sd21: lighter fallback",
        )
        sd_device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        sd_steps = st.slider(
            "Inference steps",
            min_value=10, max_value=50,
            value=30, step=5,
        )

    st.divider()
    searcher = get_searcher()
    col_a, col_b = st.columns(2)
    col_a.metric("Landmarks", searcher._index.ntotal)
    col_b.metric("Dims", "512")
    st.caption("CLIP ViT-B-16 · CPU inference · pre-computed embeddings")


# Hero
st.markdown(
    """
<div class="hero">
  <h1>🌍 Where Is This?</h1>
  <p>Upload a photo, search by description, or turn a trip into a post.</p>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="value">149</div><div class="label">Landmarks</div>
    </div>
    <div class="hero-stat">
      <div class="value">3</div><div class="label">Search modes</div>
    </div>
    <div class="hero-stat">
      <div class="value">CLIP</div><div class="label">ViT-B-16</div>
    </div>
    <div class="hero-stat">
      <div class="value">100%</div><div class="label">Golden set</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_image, tab_text, tab_story, tab_report = st.tabs(
    ["📷 Image search", "💬 Text search", "📸 Instagram post", "📊 Model report"]
)


# Tab 1: Image search
with tab_image:
    st.markdown(
        '<div class="section-label">Upload a photo of a landmark to identify it</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        col_img, col_results = st.columns([1, 2], gap="large")

        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded.name).suffix, delete=False
        ) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        photo_gps = extract_gps(str(tmp_path))

        with col_img:
            st.image(uploaded, use_container_width=True)
            if photo_gps:
                st.markdown(
                    '<div class="gps-warn">📍 <span>'
                    "This photo contains embedded GPS coordinates. "
                    "Uploading it shares your precise location with the app."
                    "</span></div>",
                    unsafe_allow_html=True,
                )

        with col_results:
            st.markdown(
                '<div class="section-label">Top matches</div>', unsafe_allow_html=True
            )
            with st.spinner("Searching…"):
                try:
                    results = searcher.search(
                        tmp_path, top_k=top_k, confidence_threshold=img_conf_threshold
                    )

                    if not results:
                        st.warning("No results found. Make sure the index is built.")
                    else:
                        if all(r.low_confidence for r in results):
                            st.warning(
                                "⚠️ All matches have low confidence - "
                                "this landmark may not be in the database."
                            )
                        for i, result in enumerate(results, start=1):
                            render_result(result, i)

                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error during search: {e}")
                    raise

        # Combined GPS map
        top_result = results[0] if results else None
        if photo_gps or (top_result and top_result.lat is not None):
            st.markdown(
                '<div class="section-label" style="margin-top:1.5rem">'
                "Location comparison</div>",
                unsafe_allow_html=True,
            )
            rows = []
            if photo_gps:
                rows.append({"lat": photo_gps[0], "lon": photo_gps[1]})
            if top_result and top_result.lat is not None:
                rows.append({"lat": top_result.lat, "lon": top_result.lon})
            if rows:
                st.map(pd.DataFrame(rows), zoom=4)
    else:
        st.markdown(
            """
        <div style="text-align:center;padding:3rem 1rem;color:#475569;">
          <div style="font-size:3rem;margin-bottom:0.75rem">📷</div>
          <div style="font-size:1rem;font-weight:600;color:#94a3b8">
            Drop an image to get started</div>
          <div style="font-size:0.85rem;margin-top:0.4rem">Supports JPG, PNG, WebP</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# Tab 2: Text search
with tab_text:
    col_input, col_mode = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "Description",
            placeholder="e.g. baroque palace with large gardens near Vienna",
            label_visibility="collapsed",
        )
    with col_mode:
        use_hybrid = st.toggle(
            "BM25 hybrid",
            value=True,
            help="Combine CLIP semantic similarity with BM25 keyword matching",
        )

    clip_weight = 0.7
    if use_hybrid:
        clip_weight = st.slider(
            "CLIP weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher = more semantic (CLIP), lower = more keyword (BM25)",
        )

    if query.strip():
        with st.spinner("Searching…"):
            try:
                if use_hybrid:
                    results = get_hybrid_searcher().search(
                        query.strip(),
                        top_k=top_k,
                        confidence_threshold=txt_conf_threshold,
                        clip_weight=clip_weight,
                    )
                else:
                    results = get_text_searcher().search(
                        query.strip(),
                        top_k=top_k,
                        confidence_threshold=txt_conf_threshold,
                    )

                if not results:
                    st.warning("No results found.")
                else:
                    if all(r.low_confidence for r in results):
                        st.warning(
                            "⚠️ Low confidence across all results - "
                            "try a more specific description."
                        )
                    if enable_image_gen:
                        st.markdown(
                            '<div class="section-label">AI visualisation vs retrieval</div>',
                            unsafe_allow_html=True,
                        )
                        col_gen, col_res = st.columns([1, 1])
                        with col_gen:
                            st.caption("🎨 What Stable Diffusion imagines")
                            with st.spinner("Generating image…"):
                                try:
                                    pipe = get_image_pipeline(model=sd_model, device=sd_device)
                                    img = generate_image(query.strip(), pipe, model=sd_model, steps=sd_steps)
                                    st.image(img, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Image generation failed: {e}")
                        with col_res:
                            st.caption("🔍 What the index retrieves")
                            for i, result in enumerate(results, start=1):
                                render_result(result, i)
                    else:
                        st.markdown(
                            '<div class="section-label">Top matches</div>',
                            unsafe_allow_html=True,
                        )
                        for i, result in enumerate(results, start=1):
                            render_result(result, i)

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error during search: {e}")
                raise

        with st.expander("🔍 Why these results? - attention visualisation"):
            st.caption(
                "Shows which tokens each attention head focuses on "
                "across all 12 layers of CLIP's text encoder. "
                "Click a coloured square to highlight that head."
            )
            with st.spinner("Generating attention map…"):
                try:
                    html = get_attention_html(query.strip())
                    components.html(html, height=600, scrolling=True)
                except Exception as e:
                    st.warning(f"Attention visualisation unavailable: {e}")
    else:
        st.markdown(
            """
        <div style="text-align:center;padding:3rem 1rem;color:#475569;">
          <div style="font-size:3rem;margin-bottom:0.75rem">💬</div>
          <div style="font-size:1rem;font-weight:600;color:#94a3b8">
            Describe a landmark to search</div>
          <div style="font-size:0.85rem;margin-top:0.4rem">
            Try: "iron tower in Paris" or "Colosseum"</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# Tab 3: Travel story
with tab_story:
    st.markdown(
        '<div class="section-label">'
        "Upload 2-5 photos from a trip to generate an Instagram post</div>",
        unsafe_allow_html=True,
    )

    uploads = st.file_uploader(
        "Choose 2-5 photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploads:
        if len(uploads) < 2:
            st.info("Add at least one more photo to generate a story.")
        elif len(uploads) > 5:
            st.warning("Please upload at most 5 photos.")
        else:
            thumb_cols = st.columns(len(uploads))
            for col, up in zip(thumb_cols, uploads):
                col.image(up, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if not ANTHROPIC_API_KEY:
                st.error(
                    "ANTHROPIC_API_KEY not set. "
                    "Add it to the .env file in the project root."
                )
            elif st.button(
                "✨ Generate Instagram post", type="primary", use_container_width=True
            ):
                identified = []
                progress = st.progress(0, text="Identifying landmarks…")

                for i, up in enumerate(uploads):
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(up.name).suffix, delete=False
                    ) as tmp:
                        tmp.write(up.getvalue())
                        tmp_path = Path(tmp.name)

                    res = searcher.search(tmp_path, top_k=1, confidence_threshold=0.0)
                    if res:
                        r = res[0]
                        identified.append(
                            {
                                "name": r.name,
                                "region": r.region,
                                "description": r.description,
                                "score": r.score,
                                "low_confidence": r.low_confidence,
                            }
                        )
                    progress.progress(
                        (i + 1) / len(uploads),
                        text=f"Identified {i + 1} of {len(uploads)}…",
                    )

                progress.empty()

                # Landmark chips
                chips_html = "".join(
                    f'<span class="lm-chip{" warn" if lm["low_confidence"] else ""}">'
                    f'{"⚠ " if lm["low_confidence"] else ""}'
                    f'{lm["name"]} · {lm["score"]:.0%}'
                    f"</span>"
                    for lm in identified
                )
                st.markdown(
                    f'<div style="margin-bottom:1rem">{chips_html}</div>',
                    unsafe_allow_html=True,
                )

                if any(lm["low_confidence"] for lm in identified):
                    st.warning(
                        "One or more landmarks have low confidence - "
                        "the post may not accurately reflect your photos."
                    )

                with st.spinner("Writing your Instagram post…"):
                    try:
                        story = generate_story(identified, api_key=ANTHROPIC_API_KEY)
                        disclaimer = (
                            "Generated by Claude using only the retrieved "
                            "landmark descriptions. Verify facts before posting."
                        )
                        st.markdown(
                            f'<div class="story-box">'
                            f'{story.replace(chr(10), "<br>")}'
                            f'<div class="story-disclaimer">{disclaimer}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Post generation failed: {e}")
    else:
        st.markdown(
            """
        <div style="text-align:center;padding:3rem 1rem;color:#475569;">
          <div style="font-size:3rem;margin-bottom:0.75rem">✈️</div>
          <div style="font-size:1rem;font-weight:600;color:#94a3b8">
            Upload photos from a trip</div>
          <div style="font-size:0.85rem;margin-top:0.4rem">
            2-5 photos · each is identified · Claude writes the post</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# Tab 4: Model Report
with tab_report:
    import json as _json

    st.markdown(
        '<div class="section-label">System performance and responsible AI audit</div>',
        unsafe_allow_html=True,
    )

    # Model info
    st.markdown("### Model")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Architecture", "CLIP ViT-B-16")
    col_m2.metric("Weights", "Fine-tuned" if CLIP_WEIGHTS_PATH else "Baseline (OpenAI)")
    col_m3.metric("Index size", f"{searcher._index.ntotal} landmarks")

    st.caption(
        "Fine-tuned on 1,184 landmark image-text pairs · "
        "last 2 transformer blocks + projection heads unfrozen (~14% of params) · "
        "InfoNCE loss · AdamW lr=5e-6 · 20 epochs · NVIDIA L40S"
        if CLIP_WEIGHTS_PATH
        else "OpenAI pretrained weights · zero-shot landmark retrieval"
    )

    st.divider()

    # Ablation table
    st.markdown("### Retrieval performance")

    baseline_path = EVAL_DIR / "ablation_baseline.json"
    finetuned_path = EVAL_DIR / "ablation_finetuned.json"

    if not baseline_path.exists() and not finetuned_path.exists():
        st.info(
            "No ablation results found. Run `python scripts/ablation.py` "
            "to generate them."
        )
    else:
        # Image retrieval table
        st.markdown("**Image retrieval - leave-one-out (148 landmarks)**")
        rows = []
        for path, label in [(baseline_path, "Baseline CLIP"), (finetuned_path, "Fine-tuned CLIP")]:
            if path.exists():
                d = _json.loads(path.read_text())
                s = d["image_summary"]
                rows.append({
                    "Condition": label,
                    "Hits@1": f"{s['hits1']}/{s['n']} ({s['hits1_pct']:.0%})",
                    "Hits@3": f"{s['hits3']}/{s['n']} ({s['hits3_pct']:.0%})",
                    "MRR": f"{s['mrr']:.3f}",
                    "Mean score": f"{s['mean_correct_score']:.3f}",
                    "95% CI": f"[{s['ci_95_lo']:.2f}, {s['ci_95_hi']:.2f}]",
                })
        if rows:
            import pandas as _pd
            st.dataframe(_pd.DataFrame(rows).set_index("Condition"), use_container_width=True)

        # Delta callout
        if baseline_path.exists() and finetuned_path.exists():
            b = _json.loads(baseline_path.read_text())["image_summary"]
            f = _json.loads(finetuned_path.read_text())["image_summary"]
            delta_h1 = f["hits1_pct"] - b["hits1_pct"]
            delta_mrr = f["mrr"] - b["mrr"]
            st.success(
                f"Fine-tuning improved Hits@1 by **{delta_h1:+.1%}** "
                f"and MRR by **{delta_mrr:+.3f}** over the baseline OpenAI weights."
            )

        # Text retrieval table
        st.markdown("**Text retrieval - strategy comparison (149 landmark name queries)**")
        text_data = None
        for path in [baseline_path, finetuned_path]:
            if path.exists():
                d = _json.loads(path.read_text())
                if d.get("text_ablation"):
                    text_data = d["text_ablation"]
                    break
        if text_data:
            clip = text_data["clip_only"]
            hybrid = text_data["hybrid"]
            text_rows = [
                {
                    "Strategy": "CLIP-only",
                    "Hits@1": f"{clip['hits1']}/{clip['n']} ({clip['hits1_pct']:.0%})",
                    "Hits@3": f"{clip['hits3']}/{clip['n']} ({clip['hits3_pct']:.0%})",
                    "MRR": f"{clip['mrr']:.3f}",
                    "95% CI": f"[{clip['ci_95_lo']:.2f}, {clip['ci_95_hi']:.2f}]",
                },
                {
                    "Strategy": "Hybrid (CLIP + BM25)",
                    "Hits@1": f"{hybrid['hits1']}/{hybrid['n']} ({hybrid['hits1_pct']:.0%})",
                    "Hits@3": f"{hybrid['hits3']}/{hybrid['n']} ({hybrid['hits3_pct']:.0%})",
                    "MRR": f"{hybrid['mrr']:.3f}",
                    "95% CI": f"[{hybrid['ci_95_lo']:.2f}, {hybrid['ci_95_hi']:.2f}]",
                },
            ]
            st.dataframe(_pd.DataFrame(text_rows).set_index("Strategy"), use_container_width=True)
            st.caption(
                "BM25 is neutral on proper-noun queries (landmark names). "
                "Its contribution appears on descriptive queries - "
                "e.g. 'ancient Roman arena' - which is its intended use case."
            )

    st.divider()

    # Geographic bias
    st.markdown("### Geographic bias audit")
    st.caption(
        "Hits@1 per region from leave-one-out evaluation. "
        "Bars below 70% are red, below 85% amber, 85%+ green."
    )
    bias_chart = EVAL_DIR / "bias_chart.png"
    if bias_chart.exists():
        st.image(str(bias_chart), use_container_width=True)
    else:
        st.info("Run `python scripts/run_audit.py` to generate the bias chart.")

    with st.expander("What does this mean?"):
        st.markdown(
            """
            The dataset skews towards **European landmarks** (Vienna + Europe = 39%).
            Regions with fewer landmarks have less training signal and fewer
            semantically distinct neighbours in the embedding space, making
            misclassification more likely.

            **Natural landmarks** (5 total) and **Oceania** (10 total) are the most
            underrepresented. A more balanced dataset - or region-specific
            fine-tuning - would reduce this gap.

            This audit is re-run whenever `scripts/run_audit.py` is executed.
            The chart above reflects the most recent run.
            """
        )

    # World map
    st.markdown("### Dataset coverage map")
    st.caption("All 149 landmarks plotted by location, coloured by region. Hover for details.")

    try:
        import pydeck as pdk

        _REGION_COLORS = {
            "Vienna":   [99,  102, 241, 220],
            "Europe":   [59,  130, 246, 220],
            "Americas": [34,  197,  94, 220],
            "Asia":     [249, 115,  22, 220],
            "Africa":   [239,  68,  68, 220],
            "Oceania":  [6,   182, 212, 220],
            "Natural":  [20,  184, 166, 220],
        }
        _DEFAULT_COLOR = [148, 163, 184, 220]

        _landmarks_all = _json.loads((DATA_DIR / "landmarks.json").read_text())
        _map_data = [
            {
                "name": lm["name"],
                "region": lm.get("region", "Unknown"),
                "lat": lm["lat"],
                "lon": lm["lon"],
                "description": lm.get("description", "")[:120] + "…",
                "color": _REGION_COLORS.get(lm.get("region", ""), _DEFAULT_COLOR),
            }
            for lm in _landmarks_all
            if lm.get("lat") and lm.get("lon")
        ]

        _layer = pdk.Layer(
            "ScatterplotLayer",
            data=_map_data,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=80000,
            pickable=True,
            auto_highlight=True,
        )

        _view = pdk.ViewState(latitude=20, longitude=10, zoom=1.2, pitch=0)

        _tooltip = {
            "html": (
                "<b>{name}</b><br/>"
                "<span style='color:#94a3b8'>{region}</span><br/>"
                "<span style='font-size:0.85em'>{description}</span>"
            ),
            "style": {
                "backgroundColor": "#1e293b",
                "color": "#e2e8f0",
                "padding": "8px 12px",
                "borderRadius": "8px",
                "maxWidth": "280px",
                "fontSize": "0.82rem",
            },
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[_layer],
                initial_view_state=_view,
                tooltip=_tooltip,
                map_style="mapbox://styles/mapbox/dark-v10",
            ),
            use_container_width=True,
            height=420,
        )

        # Region legend
        _legend_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin:3px 8px 3px 0;font-size:0.78rem;color:#cbd5e1;">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background:rgb({c[0]},{c[1]},{c[2]});flex-shrink:0"></span>'
            f'{region}</span>'
            for region, c in _REGION_COLORS.items()
        )
        st.markdown(
            f'<div style="margin-top:0.5rem">{_legend_html}</div>',
            unsafe_allow_html=True,
        )

    except Exception as _e:
        st.info(f"Map unavailable: {_e}")

    st.divider()

    # Confidence calibration
    st.markdown("### Confidence calibration")
    st.caption(
        "Score distributions for correct vs incorrect matches. "
        "The vertical line shows the configured confidence threshold (0.82)."
    )
    cal_chart = EVAL_DIR / "calibration_chart.png"
    if cal_chart.exists():
        st.image(str(cal_chart), use_container_width=True)
    else:
        st.info("Run `python scripts/run_audit.py` to generate the calibration chart.")

    col_t1, col_t2 = st.columns(2)
    col_t1.metric(
        "Image threshold", f"{CONFIDENCE_THRESHOLD_IMAGE:.2f}",
        help="Scores below this show a ⚠️ Low confidence warning",
    )
    col_t2.metric(
        "Text threshold", f"{CONFIDENCE_THRESHOLD_TEXT:.2f}",
        help="Cross-modal scores are lower due to the image-text modality gap",
    )

    st.divider()

    # Responsible AI notes
    st.markdown("### Responsible AI")
    st.markdown(
        """
        | Feature | Implementation |
        |---------|---------------|
        | **Confidence warnings** | Scores below empirical thresholds surface a ⚠️ warning instead of presenting results as certain |
        | **EXIF privacy** | Uploaded photos are checked for embedded GPS; users are warned before their location is shared |
        | **Geographic bias** | Leave-one-out audit quantifies Hits@1 per region; results shown above |
        | **Sensitive sites** | 15 landmarks flagged with access and photography restrictions; shown as yellow warnings in results |
        | **Grounded generation** | Instagram posts are grounded in retrieved Wikipedia descriptions; Claude is instructed not to invent facts |
        | **Disclaimer** | Every generated post carries a disclaimer advising fact-checking before publishing |
        """
    )
