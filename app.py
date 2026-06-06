"""
Embedding Lab — interactive exploration of word-vector arithmetic.

Streamlit application that uses GloVe-Wiki-Gigaword-100 embeddings and
cosine similarity to build additive/subtractive expressions, find nearest
neighbors, and visualize vectors in a 3D PCA projection.

Run:
    streamlit run embeddings_intuition.py

Dependencies:
    streamlit, gensim, numpy, plotly, scikit-learn
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Embedding Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', -apple-system, sans-serif !important;
    letter-spacing: -0.5px !important;
}
.stApp { 
    background-color: #0f1117 !important; 
    color: #e1e2e8 !important; 
}
.block-container {
    padding-top: 2rem !important;
    max-width: 1000px;
    margin: auto;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

header[data-testid="stHeader"]  { display: none !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

/* Build Expression */
.st-key-build_expr_container > div[data-testid="stVerticalBlockBorderWrapper"],
.stVerticalBlock.st-key-build_expr_container,
div[data-testid="stVerticalBlock"].st-key-build_expr_container {
    background-color: #1a2a3d !important;
    border: 2px solid #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55), 0 2px 10px rgba(96, 165, 250, 0.15) !important;
}

/* Closest Words */
.st-key-closest_words_container > div[data-testid="stVerticalBlockBorderWrapper"],
.stVerticalBlock.st-key-closest_words_container,
div[data-testid="stVerticalBlock"].st-key-closest_words_container {
    background-color: #2e2210 !important;
    border: 2px solid #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55), 0 2px 10px rgba(251, 191, 36, 0.15) !important;
}

/* 3D Visualization */
.st-key-viz_3d_container > div[data-testid="stVerticalBlockBorderWrapper"],
.stVerticalBlock.st-key-viz_3d_container,
div[data-testid="stVerticalBlock"].st-key-viz_3d_container {
    background-color: #131a2b !important;
    border: 2px solid #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55), 0 2px 10px rgba(96, 165, 250, 0.15) !important;
}

/* Inputs and selectboxes */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div > div {
    color: #e1e2e8 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease !important;
    outline: none !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextInput"] input:focus-visible,
div[data-testid="stTextInput"] input:active {
    outline: none !important;
}
/* Input outlines */
div[data-testid="stTextInput"] > div {
    border-color: transparent !important;
}
div[data-testid="stTextInput"] > div:focus-within {
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Build Expression inputs */
.st-key-build_expr_container div[data-testid="stTextInput"] input,
.st-key-build_expr_container div[data-testid="stSelectbox"] > div > div {
    background-color: #1a1b26 !important;
    border: 1px solid #3a3d4e !important;
    outline: none !important;
}
.st-key-build_expr_container div[data-testid="stTextInput"] input:focus-visible,
.st-key-build_expr_container div[data-testid="stTextInput"] input:focus {
    outline: none !important;
}

/* Term input */
.st-key-word_input_main div[data-testid="stTextInput"] input {
    caret-color: #ffffff !important;
    caret-width: thick !important;
    font-weight: 600 !important;
}

/* Saved Vectors selectbox — visible @ ref while typing and after selection */
.st-key-saved_pick_main div[data-testid="stSelectbox"] [data-baseweb="select"] input {
    caret-color: #ffffff !important;
    caret-width: thick !important;
    font-weight: 600 !important;
    color: #e1e2e8 !important;
    -webkit-text-fill-color: #e1e2e8 !important;
    opacity: 1 !important;
}
.st-key-saved_pick_main div[data-testid="stSelectbox"] [class*="singleValue"] {
    font-weight: 600 !important;
    color: #e1e2e8 !important;
}
.st-key-saved_pick_main div[data-testid="stSelectbox"] > div:focus-within > div {
    border-color: #8b8fa3 !important;
    box-shadow: 0 0 0 1px #8b8fa3 !important;
}

/* Closest Words inputs */
.st-key-closest_words_container div[data-testid="stTextInput"] input {
    background-color: #141008 !important;
    border: 1px solid #78552b !important;
}

/* Build Expression input focus */
.st-key-build_expr_container div[data-testid="stTextInput"] input:focus {
    border-color: #8b8fa3 !important;
    box-shadow: 0 0 0 1px #8b8fa3 !important;
}
/* Closest Words input focus */
.st-key-closest_words_container div[data-testid="stTextInput"] input:focus {
    border-color: #fbbf24 !important;
    box-shadow: 0 0 0 1px #fbbf24 !important;
}
/* Input focus fallback */
div[data-testid="stTextInput"] input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 1px #818cf8 !important;
}
div[data-testid="stTextInput"] label, div[data-testid="stSelectbox"] label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #8b8fa3 !important;
}

/* Buttons */
div[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    background-color: transparent !important;
    color: #e1e2e8 !important;
    padding: 8px 16px !important;
    width: 100%;
}

/* Build Expression buttons */
.st-key-build_expr_container div[data-testid="stButton"] > button {
    border: 1px solid #166552 !important;
}
.st-key-build_expr_container div[data-testid="stButton"] > button:hover {
    border-color: #1d8a6d !important;
    background-color: #0d2a22 !important;
}

/* Closest Words buttons */
.st-key-closest_words_container div[data-testid="stButton"] > button {
    border: 1px solid #78552b !important;
}
.st-key-closest_words_container div[data-testid="stButton"] > button:hover {
    border-color: #a0723a !important;
    background-color: #221c0f !important;
}
/* Build Expression primary button */
.st-key-build_expr_container div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #10b981 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
}
.st-key-build_expr_container div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #059669 !important;
}
/* Primary button fallback */
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #818cf8 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #6366f1 !important;
}
div[data-testid="stButton"] > button p,
div[data-testid="stButton"] > button span,
div[data-testid="stButton"] > button div {
    color: inherit !important;
}

/* Add button */
.st-key-add_word div[data-testid="stButton"] button {
    background-color: #34d399 !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}
.st-key-add_word div[data-testid="stButton"] button:hover {
    background-color: #10b981 !important;
}
.st-key-add_word div[data-testid="stButton"] button p,
.st-key-add_word div[data-testid="stButton"] button span,
.st-key-add_word div[data-testid="stButton"] button div {
    color: #000000 !important;
}

/* Sub button */
.st-key-sub_word div[data-testid="stButton"] button {
    background-color: #f87171 !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}
.st-key-sub_word div[data-testid="stButton"] button:hover {
    background-color: #ef4444 !important;
}
.st-key-sub_word div[data-testid="stButton"] button p,
.st-key-sub_word div[data-testid="stButton"] button span,
.st-key-sub_word div[data-testid="stButton"] button div {
    color: #000000 !important;
}

/* Clear button */
.st-key-clear_expr div[data-testid="stButton"] button {
    background-color: #fb923c !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}
.st-key-clear_expr div[data-testid="stButton"] button:hover {
    background-color: #f97316 !important;
}
.st-key-clear_expr div[data-testid="stButton"] button p,
.st-key-clear_expr div[data-testid="stButton"] button span,
.st-key-clear_expr div[data-testid="stButton"] button div {
    color: #000000 !important;
}

/* Chip remove buttons */
.st-key-chips_add div[data-testid="stButton"] > button,
.st-key-chips_sub div[data-testid="stButton"] > button {
    width: auto !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 3px 12px !important;
    white-space: nowrap !important;
    border-left-width: 1px !important;
}
.st-key-chips_add div[data-testid="stButton"] > button {
    background-color: rgba(110, 231, 183, 0.08) !important;
    color: #6ee7b7 !important;
    border: 1px solid rgba(110, 231, 183, 0.25) !important;
}
.st-key-chips_add div[data-testid="stButton"] > button:hover {
    background-color: rgba(110, 231, 183, 0.18) !important;
    border-color: rgba(110, 231, 183, 0.5) !important;
}
.st-key-chips_sub div[data-testid="stButton"] > button {
    background-color: rgba(252, 165, 165, 0.08) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(252, 165, 165, 0.25) !important;
}
.st-key-chips_sub div[data-testid="stButton"] > button:hover {
    background-color: rgba(252, 165, 165, 0.18) !important;
    border-color: rgba(252, 165, 165, 0.5) !important;
}

/* Progress */
.stSpinner > div > div { border-top-color: #10b981 !important; }
.stProgress > div > div > div > div { background-color: #10b981 !important; }

/* Typography */
.title-main {
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.title-sub {
    font-size: 15px;
    color: #8b8fa3;
    margin-bottom: 24px;
    line-height: 1.5;
}

.section-title {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 600;
    margin: 0 0 4px 0;
    color: #e1e2e8;
}
/* Build Expression title */
.st-key-build_expr_container .section-title {
    color: #34d399;
}
/* Closest Words title */
.st-key-closest_words_container .section-title {
    color: #fbbf24;
}
.section-sub {
    font-size: 13px;
    color: #8b8fa3;
    margin-bottom: 16px;
    line-height: 1.5;
}

/* Expression box */
.expr-box {
    background-color: #1a1b26;
    border: 1px solid #3a3d4e;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}

/* Closest Words expression box */
.st-key-closest_words_container .expr-box {
    background-color: #141008;
    border: 1px solid #78552b;
}
.expr-formula {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    color: #e1e2e8;
    word-break: break-word;
}
.expr-empty {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #565a6e;
    font-style: italic;
}

/* Chips */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.word-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}
.chip-add {
    background-color: rgba(110, 231, 183, 0.1);
    color: #6ee7b7;
    border: 1px solid rgba(110, 231, 183, 0.2);
}
.chip-sub {
    background-color: rgba(252, 165, 165, 0.1);
    color: #fca5a5;
    border: 1px solid rgba(252, 165, 165, 0.2);
}
.chip-saved {
    background-color: rgba(251, 191, 36, 0.1);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.2);
}

/* Stepper */
.stepper {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
    font-size: 13px;
    color: #8b8fa3;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: #166552;
    color: #e1e2e8;
    font-size: 11px;
    font-weight: 600;
}
.step-divider {
    height: 1px;
    width: 30px;
    background-color: #166552;
}

/* Results */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
.neighbor-card {
    background-color: #141008;
    border-left: 3px solid #fbbf24;
    border-top: 1px solid #78552b;
    border-right: 1px solid #78552b;
    border-bottom: 1px solid #78552b;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    animation: fadeIn 0.3s ease-out;
}
.neighbor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.neighbor-rank {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #565a6e;
    margin-right: 8px;
}
.neighbor-word {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #e1e2e8;
}
.neighbor-score {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #8b8fa3;
}
.neighbor-bar-bg {
    background-color: #1a150a;
    border-radius: 4px;
    height: 4px;
    overflow: hidden;
}
.neighbor-bar-fill {
    height: 100%;
    background-color: #fbbf24;
    border-radius: 4px;
}

/* History */
.history-row {
    background-color: #0f1117;
    border: 1px solid #2e3258;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.history-expr {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #fbbf24;
}
.history-res {
    font-size: 12px;
    color: #8b8fa3;
    margin-top: 4px;
}

/* History buttons */
.st-key-experiment_history div[data-testid="stButton"] > button {
    border: 2px solid #8b8fa3 !important;
}
.st-key-experiment_history div[data-testid="stButton"] > button:hover {
    border-color: #e1e2e8 !important;
    background-color: #1a1b26 !important;
}

/* Expander */
.streamlit-expanderHeader,
div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] details summary p {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    color: #8b8fa3 !important;
    background-color: transparent !important;
}
.streamlit-expanderHeader:hover,
.streamlit-expanderHeader:focus,
.streamlit-expanderHeader:active,
div[data-testid="stExpander"] details summary:hover,
div[data-testid="stExpander"] details summary:focus,
div[data-testid="stExpander"] details summary:active,
div[data-testid="stExpander"] details summary:hover p {
    background-color: transparent !important;
    color: #e1e2e8 !important;
}
div[data-testid="stExpander"], 
div[data-testid="stExpander"] details, 
div[data-testid="stExpander"] details > div,
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: transparent !important;
}

</style>
<script>
    // Disable autocomplete on text inputs across initial load and rerenders.
    document.addEventListener('DOMContentLoaded', function() {
        const inputs = document.querySelectorAll('input[type="text"]');
        inputs.forEach(input => {
            input.setAttribute('autocomplete', 'off');
        });
    });
    const observer = new MutationObserver(function(mutations) {
        const inputs = document.querySelectorAll('input[type="text"]');
        inputs.forEach(input => {
            input.setAttribute('autocomplete', 'off');
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────
TOP_NEIGHBORS = 4

# ─── Model loading ─────────────────────────────────────────────────────────
MODEL_NAME = "glove-wiki-gigaword-100"

def is_model_downloaded():
    """Return True if the GloVe model is present in the gensim data directory."""
    import os
    from gensim import downloader as gd
    return os.path.exists(os.path.join(gd.BASE_DIR, MODEL_NAME))

def download_model_with_progress(progress_bar, status_label):
    """Download the embedding model and stream progress to Streamlit widgets."""
    import gensim.downloader as gd

    class _StreamlitDownloadProgress:
        def __call__(self, chunks_downloaded, chunk_size, total_size, part=1, total_parts=1):
            if total_size <= 0:
                return
            part_frac = min((chunks_downloaded * chunk_size) / total_size, 1.0)
            overall = ((part - 1) + part_frac) / total_parts
            progress_bar.progress(overall)
            downloaded_mb = (chunks_downloaded * chunk_size) / 1e6
            total_mb = total_size / 1e6
            part_note = f" · part {part}/{total_parts}" if total_parts > 1 else ""
            status_label.markdown(
                f"Downloading … **{downloaded_mb:.0f} / {total_mb:.0f} MB**{part_note}"
            )

    original_progress = gd._progress
    gd._progress = _StreamlitDownloadProgress()
    try:
        gd._download(MODEL_NAME)
    finally:
        gd._progress = original_progress
    progress_bar.progress(1.0)

def load_vectors_with_progress(load_bar, status_label):
    """Load word vectors from disk into a gensim KeyedVectors model."""
    import os
    from gensim import downloader as gd
    from gensim import utils
    from gensim.models import KeyedVectors
    from gensim.models import keyedvectors as kv_mod

    path = os.path.join(gd.BASE_DIR, MODEL_NAME, f"{MODEL_NAME}.gz")
    with utils.open(path, "rb") as fin:
        header = utils.to_unicode(fin.readline(), encoding="utf8")
        vocab_size, vector_size = [int(x) for x in header.split()]
        kv = KeyedVectors(vector_size, vocab_size)
        report_every = max(1, vocab_size // 100)

        for line_no in range(vocab_size):
            line = fin.readline()
            if line == b"":
                raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
            word, weights = kv_mod._word2vec_line_to_vector(line, kv_mod.REAL, "strict", "utf8")
            kv_mod._add_word_to_kv(kv, None, word, weights, vocab_size)
            if line_no % report_every == 0 or line_no == vocab_size - 1:
                pct = (line_no + 1) / vocab_size
                load_bar.progress(pct)
                status_label.markdown(
                    f"Loading into memory … **{line_no + 1:,} / {vocab_size:,} vectors**"
                )

    if kv.vectors.shape[0] != len(kv):
        kv.vectors = kv_mod.ascontiguousarray(kv.vectors[: len(kv)])
    load_bar.progress(1.0)
    return kv

def load_model_with_progress(download_bar, load_bar, status_label, show_download):
    """Download the model when required, then load vectors into memory."""
    if show_download:
        download_model_with_progress(download_bar, status_label)
    return load_vectors_with_progress(load_bar, status_label)

# ─── Expression helpers ─────────────────────────────────────────────────────
def is_saved_ref(term):
    """Return True if term is a saved-vector reference (e.g. '@v1')."""
    return isinstance(term, str) and term.startswith("@")

def term_label(term, saved_vectors):
    """Return the display label for a vocabulary word or saved-vector reference."""
    if is_saved_ref(term):
        vid = term[1:]
        return saved_vectors.get(vid, {}).get("label", term)
    return term

def format_expression(add_terms, sub_terms, saved_vectors):
    """Format add/subtract terms as a readable expression string."""
    if not add_terms and not sub_terms:
        return "∅"

    if not sub_terms:
        return " + ".join(term_label(t, saved_vectors) for t in add_terms)

    if not add_terms:
        return " ".join(f"- {term_label(t, saved_vectors)}" for t in sub_terms)

    # Group as (first_add - sub_terms) + remaining_add_terms.
    first_add = term_label(add_terms[0], saved_vectors)
    sub_part = " - ".join(term_label(t, saved_vectors) for t in sub_terms)
    inner = f"({first_add} - {sub_part})"
    if len(add_terms) == 1:
        return inner
    rest = " + ".join(term_label(t, saved_vectors) for t in add_terms[1:])
    return f"{inner} + {rest}"

def resolve_term(term, model, saved_vectors):
    """Resolve a term to its embedding vector, or return an error indicator."""
    if is_saved_ref(term):
        vid = term[1:]
        if vid not in saved_vectors:
            return None, f"saved vector '{term}' not found"
        return saved_vectors[vid]["vector"].copy(), None
    if term not in model:
        return None, term
    return model[term].copy(), None

def vector_from_expression(add_terms, sub_terms, model, saved_vectors):
    """Compute the vector for an add/subtract expression."""
    if not add_terms and not sub_terms:
        return None, ["expression is empty"]
    vec = np.zeros(model.vector_size)
    missing = []
    for t in add_terms:
        v, err = resolve_term(t, model, saved_vectors)
        if err:
            missing.append(err)
        else:
            vec += v
    for t in sub_terms:
        v, err = resolve_term(t, model, saved_vectors)
        if err:
            missing.append(err)
        else:
            vec -= v
    if missing:
        return None, missing
    return vec, []

def nearest_words_for_vec(vec, model, exclude_words=None, topn=TOP_NEIGHBORS):
    """Return the top cosine-similar vocabulary words for a vector."""
    buffer = topn + len(exclude_words or []) + 20
    candidates = model.similar_by_vector(vec, topn=buffer)
    if exclude_words:
        exclude_set = set(exclude_words)
        candidates = [(w, s) for w, s in candidates if w not in exclude_set]
    return candidates[:topn]

# ─── 3D visualization ──────────────────────────────────────────────────────
def _grid_tick_step(span, target_count=6):
    """Choose a readable tick interval for a scene axis span."""
    if span <= 0:
        return 1.0
    rough = span / target_count
    exponent = 10 ** np.floor(np.log10(rough))
    for factor in (1, 2, 5, 10):
        step = factor * exponent
        if step >= rough * 0.5:
            return float(step)
    return float(10 * exponent)


def _grid_ticks(lo, hi, step):
    """Return tick positions between lo and hi at the given step size."""
    start = np.ceil(lo / step) * step
    return [float(t) for t in np.arange(start, hi + step * 0.5, step) if lo - 1e-9 <= t <= hi + 1e-9]


def _scene_grid_bounds(coords_3d, padding=0.15):
    """Return padded lower and upper bounds for the 3D scene axes."""
    mins = coords_3d.min(axis=0)
    maxs = coords_3d.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = spans * padding
    return mins - pad, maxs + pad


def _add_scene_box_grid(fig, lo, hi, color='#7a8299', width=1.2, divisions=6):
    """
    Draw crossed grid lines on the bottom, back, and side planes of the scene box.

    Plotly's built-in 3D grid only crosses on one plane; custom line traces
    provide a consistent grid on all three visible faces.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi

    x_ticks = _grid_ticks(x0, x1, _grid_tick_step(x1 - x0, divisions))
    y_ticks = _grid_ticks(y0, y1, _grid_tick_step(y1 - y0, divisions))
    z_ticks = _grid_ticks(z0, z1, _grid_tick_step(z1 - z0, divisions))

    xs, ys, zs = [], [], []

    def segment(ax, ay, az, bx, by, bz):
        xs.extend([ax, bx, None])
        ys.extend([ay, by, None])
        zs.extend([az, bz, None])

    # Bottom plane (z = z0)
    for y in y_ticks:
        segment(x0, y, z0, x1, y, z0)
    for x in x_ticks:
        segment(x, y0, z0, x, y1, z0)

    # Back plane (y = y0)
    for z in z_ticks:
        segment(x0, y0, z, x1, y0, z)
    for x in x_ticks:
        segment(x, y0, z0, x, y0, z1)

    # Side plane (x = x0)
    for z in z_ticks:
        segment(x0, y0, z, x0, y1, z)
    for y in y_ticks:
        segment(x0, y, z0, x0, y, z1)

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=color, width=width),
        hoverinfo='skip',
        showlegend=False,
    ))


def create_3d_visualization(add_terms, sub_terms, model, saved_vectors, result=None):
    """
    Build a 3D Plotly figure for the current vector expression.

    Projects word vectors onto three PCA components, anchors the origin at
    (0, 0, 0), and draws expression arrows with a scene-box grid.

    Args:
        add_terms: Terms to add in the expression.
        sub_terms: Terms to subtract in the expression.
        model: Loaded gensim KeyedVectors model.
        saved_vectors: Saved intermediate vectors keyed by id.
        result: Optional computed result dict with vector and nearest neighbors.
            When omitted, only cumulative add/subtract steps are shown.

    Returns:
        A Plotly Figure, or None when fewer than two points are available.
    """
    computed_mode = result is not None

    all_vectors = []
    all_labels = []
    all_types = []

    origin_vec = np.zeros(model.vector_size)
    all_vectors.append(origin_vec)
    all_labels.append("Origin")
    all_types.append('origin')
    
    cumulative_positions = []
    cumulative_vec = np.zeros(model.vector_size)

    for t in add_terms:
        vec, err = resolve_term(t, model, saved_vectors)
        if not err:
            cumulative_vec = cumulative_vec + vec
            cumulative_positions.append((cumulative_vec.copy(), f"+{term_label(t, saved_vectors)}", 'add'))
    
    for t in sub_terms:
        vec, err = resolve_term(t, model, saved_vectors)
        if not err:
            cumulative_vec = cumulative_vec - vec
            cumulative_positions.append((cumulative_vec.copy(), f"−{term_label(t, saved_vectors)}", 'sub'))
    
    if computed_mode:
        all_vectors.append(result["vector"])
        all_labels.append("Result")
        all_types.append('result')
        
        for word, score in result["nearest"][:TOP_NEIGHBORS]:
            if word in model:
                all_vectors.append(model[word])
                all_labels.append(f"{word} ({score:.2f})")
                all_types.append('neighbor')
    else:
        for cum_vec, label, op_type in cumulative_positions:
            all_vectors.append(cum_vec)
            all_labels.append(label)
            all_types.append(op_type)
    
    if len(all_vectors) < 2:
        return None

    vectors_array = np.array(all_vectors)
    n_components = min(3, len(all_vectors))
    pca = PCA(n_components=n_components)
    coords_3d = pca.fit_transform(vectors_array)
    
    if n_components < 3:
        padding = np.zeros((coords_3d.shape[0], 3 - n_components))
        coords_3d = np.hstack([coords_3d, padding])

    # Anchor the origin at (0, 0, 0) in the projected space.
    origin_offset = coords_3d[0].copy()
    coords_3d = coords_3d - origin_offset

    # Size the scene grid from all displayed points, including the build path.
    grid_points = [coords_3d]
    if computed_mode:
        path_vectors = [np.zeros(model.vector_size)]
        path_cumulative = np.zeros(model.vector_size)
        for t in add_terms:
            vec, err = resolve_term(t, model, saved_vectors)
            if not err:
                path_cumulative = path_cumulative + vec
                path_vectors.append(path_cumulative.copy())
        for t in sub_terms:
            vec, err = resolve_term(t, model, saved_vectors)
            if not err:
                path_cumulative = path_cumulative - vec
                path_vectors.append(path_cumulative.copy())
        if len(path_vectors) > 1:
            path_coords = pca.transform(np.array(path_vectors))
            if path_coords.shape[1] < 3:
                pad = np.zeros((path_coords.shape[0], 3 - path_coords.shape[1]))
                path_coords = np.hstack([path_coords, pad])
            grid_points.append(path_coords - origin_offset)

    grid_lo, grid_hi = _scene_grid_bounds(np.vstack(grid_points))

    fig = go.Figure()
    _add_scene_box_grid(fig, grid_lo, grid_hi, width=1.2)
    legend_shown = {'add': False, 'sub': False}

    color_map = {
        'origin': '#6b7280',
        'add': '#34d399',
        'sub': '#f87171',
        'result': '#fbbf24',
        'neighbor': '#fbbf24',
    }
    
    def add_arrow_with_cone(fig, start, end, color, label, show_label=True, line_width=5, cone_scale=1.0,
                            arrow_type=None):
        """Draw a 3D arrow from start to end using a line segment and cone tip."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if length < 1e-10:
            return

        ux, uy, uz = dx/length, dy/length, dz/length
        cone_size = max(length * 0.12 * cone_scale, 0.2 * cone_scale)
        shaft_end = [end[0] - ux * cone_size * 0.3, 
                     end[1] - uy * cone_size * 0.3, 
                     end[2] - uz * cone_size * 0.3]

        legendgroup = None
        legend_name = None
        show_in_legend = False
        if arrow_type == 'add':
            legendgroup = 'add_arrows'
            legend_name = 'Add (+)'
            show_in_legend = not legend_shown['add']
            legend_shown['add'] = True
        elif arrow_type == 'sub':
            legendgroup = 'sub_arrows'
            legend_name = 'Subtract (−)'
            show_in_legend = not legend_shown['sub']
            legend_shown['sub'] = True

        line_color = color_map.get(arrow_type, color) if show_in_legend else color
        
        fig.add_trace(go.Scatter3d(
            x=[start[0], shaft_end[0]],
            y=[start[1], shaft_end[1]],
            z=[start[2], shaft_end[2]],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            legendgroup=legendgroup,
            name=legend_name,
            showlegend=show_in_legend,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Cone(
            x=[end[0]], y=[end[1]], z=[end[2]],
            u=[ux], v=[uy], w=[uz],
            sizemode='absolute',
            sizeref=cone_size,
            anchor='tip',
            colorscale=[[0, color], [1, color]],
            showscale=False,
            legendgroup=legendgroup,
            name=legend_name,
            showlegend=False,
            hoverinfo='skip'
        ))

        if show_label and label:
            if arrow_type == 'add':
                label_color = color_map['add']
            elif arrow_type == 'sub':
                label_color = color_map['sub']
            else:
                label_color = '#e1e2e8'
            fig.add_trace(go.Scatter3d(
                x=[end[0]], y=[end[1]], z=[end[2]],
                mode='text',
                text=[label],
                textposition='top center',
                textfont=dict(size=11, color=label_color),
                legendgroup=legendgroup,
                name=legend_name,
                showlegend=False,
                hovertemplate=f'<b>{label}</b><extra></extra>'
            ))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=10, color='#ffffff', symbol='circle'),
        text=['(0, 0, 0)'],
        textposition='top center',
        textfont=dict(size=10, color='#ffffff'),
        name='Origin',
        hovertemplate='<b>Origin</b> (0, 0, 0)<extra></extra>'
    ))
    
    if computed_mode:
        # Overlay the build path, result vector, and nearest neighbors.
        build_vectors = [origin_vec]
        build_labels = ['Origin']
        build_types = ['origin']
        cumulative = np.zeros(model.vector_size)
        
        for t in add_terms:
            vec, err = resolve_term(t, model, saved_vectors)
            if not err:
                cumulative = cumulative + vec
                build_vectors.append(cumulative.copy())
                build_labels.append(f"+{term_label(t, saved_vectors)}")
                build_types.append('add')
        
        for t in sub_terms:
            vec, err = resolve_term(t, model, saved_vectors)
            if not err:
                cumulative = cumulative - vec
                build_vectors.append(cumulative.copy())
                build_labels.append(f"−{term_label(t, saved_vectors)}")
                build_types.append('sub')
        
        if len(build_vectors) > 1:
            build_array = np.array(build_vectors)
            build_coords = pca.transform(build_array)
            if build_coords.shape[1] < 3:
                pad = np.zeros((build_coords.shape[0], 3 - build_coords.shape[1]))
                build_coords = np.hstack([build_coords, pad])
            build_coords = build_coords - origin_offset

            path_colors = {
                'add': 'rgba(52, 211, 153, 0.4)',
                'sub': 'rgba(248, 113, 113, 0.4)',
            }

            prev_idx = 0
            for i in range(1, len(build_types)):
                point_type = build_types[i]
                arrow_color = path_colors.get(point_type, '#6b7280')
                label = build_labels[i]

                start = [build_coords[prev_idx, 0], build_coords[prev_idx, 1], build_coords[prev_idx, 2]]
                end = [build_coords[i, 0], build_coords[i, 1], build_coords[i, 2]]

                add_arrow_with_cone(
                    fig, start, end, arrow_color, label,
                    line_width=3, cone_scale=0.7, arrow_type=point_type,
                )
                
                prev_idx = i

        result_idx = all_types.index('result')
        start = [0, 0, 0]
        end = [coords_3d[result_idx, 0], coords_3d[result_idx, 1], coords_3d[result_idx, 2]]
        add_arrow_with_cone(fig, start, end, color_map['result'], 'Result', line_width=5, cone_scale=0.9)

        neighbor_indices = [i for i, t in enumerate(all_types) if t == 'neighbor']
        if neighbor_indices:
            x = [coords_3d[i, 0] for i in neighbor_indices]
            y = [coords_3d[i, 1] for i in neighbor_indices]
            z = [coords_3d[i, 2] for i in neighbor_indices]
            text = [all_labels[i] for i in neighbor_indices]

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(size=7, color=color_map['neighbor'], symbol='circle', 
                           line=dict(color='#92400e', width=1)),
                text=text,
                textposition='top center',
                textfont=dict(size=9, color='#d97706'),
                name='Neighbors',
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
    else:
        prev_idx = 0
        for i in range(1, len(all_types)):
            point_type = all_types[i]
            arrow_color = color_map[point_type]
            label = all_labels[i]
            
            start = [coords_3d[prev_idx, 0], coords_3d[prev_idx, 1], coords_3d[prev_idx, 2]]
            end = [coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2]]
            
            add_arrow_with_cone(fig, start, end, arrow_color, label, arrow_type=point_type)
            
            prev_idx = i

    scene_axis = dict(
        showgrid=False,
        showticklabels=False,
        title='',
        backgroundcolor='#0f1117',
        zeroline=True,
        zerolinecolor='#9aa0b8',
        zerolinewidth=1.5,
        showline=True,
        linecolor='#6b7280',
        linewidth=1.5,
    )
    fig.update_layout(
        scene=dict(
            xaxis={**scene_axis, 'range': [grid_lo[0], grid_hi[0]]},
            yaxis={**scene_axis, 'range': [grid_lo[1], grid_hi[1]]},
            zaxis={**scene_axis, 'range': [grid_lo[2], grid_hi[2]]},
            bgcolor='#0f1117',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor='#0f1117',
        plot_bgcolor='#0f1117',
        font=dict(color='#e1e2e8'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=12, family='Inter, sans-serif', weight='bold'),
        ),
        showlegend=True,
        uirevision='constant',
    )
    
    return fig

# ─── Session state ─────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "expr": {"add": [], "sub": []},
        "saved_vectors": {},
        "history": [],
        "result": None,
        "model_loaded": False,
        "wv": None,
        "next_cell": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

def save_vector_from_result(label, vector):
    """Persist a computed vector in session state and return its reference id."""
    vid = f"v{len(st.session_state.saved_vectors) + 1}"
    st.session_state.saved_vectors[vid] = {
        "label": label,
        "vector": vector.copy(),
    }
    return vid

def append_history(expr_str, nearest, vector, add_terms, sub_terms):
    """Record a computed expression and its result in session history."""
    cell_id = st.session_state.next_cell
    st.session_state.next_cell += 1
    vid = save_vector_from_result(expr_str, vector)
    st.session_state.history.insert(0, {
        "id": cell_id,
        "expr_str": expr_str,
        "nearest": nearest,
        "vector_ref": f"@{vid}",
        "add": list(add_terms),
        "sub": list(sub_terms),
    })

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div>
  <div class="title-main">Embedding Lab</div>
  <div class="title-sub">Explore semantic relationships through vector arithmetic in a high-dimensional space.</div>
</div>

<div class="stepper">
  <div class="step-item">
    <div class="step-num">1</div>
    <span>Build expression</span>
  </div>
  <div class="step-divider"></div>
  <div class="step-item">
    <div class="step-num">2</div>
    <span>Compute</span>
  </div>
  <div class="step-divider"></div>
  <div class="step-item">
    <div class="step-num">3</div>
    <span>Discover</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Model loader ───────────────────────────────────────────────────────────
if not st.session_state.model_loaded:
    st.markdown("""
    <div style='background-color:#1a1b26; border:1px solid #2a2d3e; border-radius:8px;
         padding:20px 24px; margin-bottom:20px; max-width:600px'>
      <div style='font-size:12px; letter-spacing:1px; color:#e1e2e8; font-weight:600;
           margin-bottom:8px; text-transform:uppercase;'>Model Required</div>
      <div style='font-size:13px; color:#8b8fa3; line-height:1.6'>
        This lab uses <strong>GloVe-Wiki-Gigaword-100</strong>.<br>
        Initial download is ~134 MB. Subsequent runs load from local cache.
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load Model", type="primary"):
        try:
            need_download = not is_model_downloaded()
            status_label = st.empty()

            download_bar = None
            if need_download:
                st.caption("Downloading Model")
                download_bar = st.progress(0.0)
                status_label.markdown("Preparing download …")

            st.caption("Loading into Memory")
            load_bar = st.progress(0.0)

            wv = load_model_with_progress(
                download_bar, load_bar, status_label, show_download=need_download
            )
            status_label.markdown("**Model ready.**")
            st.session_state.wv = wv
            st.session_state.model_loaded = True
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    st.stop()

wv = st.session_state.wv
expr = st.session_state.expr
saved = st.session_state.saved_vectors

# ─── Main workspace ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    with st.container(border=True, key="build_expr_container"):
        st.markdown(
            '<div class="section-title">Build Expression</div>'
            '<div class="section-sub">Add or subtract terms to navigate the semantic space.</div>',
            unsafe_allow_html=True,
        )

        word_input = st.text_input("Term", placeholder="e.g. king", key="word_input_main")
        word = word_input.strip().lower()

        picked_ref = None
        if saved:
            saved_options = [
                f"@{vid} ({meta['label'][:40]})" for vid, meta in saved.items()
            ]
            picked_saved = st.selectbox(
                "Saved Vectors",
                saved_options,
                index=None,
                placeholder="— reuse a past result —",
                key="saved_pick_main",
            )
            picked_ref = picked_saved.split()[0] if picked_saved else None

        with st.container(key="expr_btn_row"):
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                add_clicked = st.button("＋ Add", key="add_word")
            with b2:
                sub_clicked = st.button("− Sub", key="sub_word")
            with b3:
                clear_clicked = st.button("Clear", key="clear_expr")
            with b4:
                compute_clicked = st.button("Compute", type="primary", key="compute_btn")

        if compute_clicked:
            vec, missing = vector_from_expression(expr["add"], expr["sub"], wv, saved)
            if missing:
                st.error(f"Could not compute — missing: {', '.join(missing)}")
            else:
                exclude = [t for t in expr["add"] + expr["sub"] if not is_saved_ref(t)]
                nearest = nearest_words_for_vec(vec, wv, exclude_words=exclude)
                expr_str = format_expression(expr["add"], expr["sub"], saved)
                st.session_state.result = {
                    "expr_str": expr_str,
                    "nearest": nearest,
                    "vector": vec,
                }
                append_history(expr_str, nearest, vec, expr["add"], expr["sub"])
                st.rerun()

        if add_clicked:
            if word:
                if any(c.isdigit() for c in word):
                    st.warning("Numbers are not allowed.")
                elif word not in wv:
                    st.warning(f"'{word}' not in vocabulary.")
                elif word not in expr["add"]:
                    expr["add"].append(word)
                    st.rerun()
            elif picked_ref and picked_ref not in expr["add"]:
                expr["add"].append(picked_ref)
                st.rerun()

        if sub_clicked:
            if word:
                if any(c.isdigit() for c in word):
                    st.warning("Numbers are not allowed.")
                elif word not in wv:
                    st.warning(f"'{word}' not in vocabulary.")
                elif word not in expr["sub"]:
                    expr["sub"].append(word)
                    st.rerun()
            elif picked_ref and picked_ref not in expr["sub"]:
                expr["sub"].append(picked_ref)
                st.rerun()

        if clear_clicked:
            st.session_state.expr = {"add": [], "sub": []}
            st.session_state.result = None
            st.rerun()

        expr_str = format_expression(expr["add"], expr["sub"], saved)

        if not expr["add"] and not expr["sub"]:
            st.markdown(
                '<div class="expr-box"><span class="expr-empty">Expression is empty. Add a term to begin.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="expr-box"><div class="expr-formula">{expr_str}</div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="chips_add"):
                if expr["add"]:
                    cols = st.columns(len(expr["add"]))
                    for col, t in zip(cols, list(expr["add"])):
                        lbl = term_label(t, saved)
                        with col:
                            if st.button(f"+ {lbl}  ✕", key=f"chip_rm_add_{t}"):
                                expr["add"].remove(t)
                                st.rerun()
            with st.container(key="chips_sub"):
                if expr["sub"]:
                    cols = st.columns(len(expr["sub"]))
                    for col, t in zip(cols, list(expr["sub"])):
                        lbl = term_label(t, saved)
                        with col:
                            if st.button(f"− {lbl}  ✕", key=f"chip_rm_sub_{t}"):
                                expr["sub"].remove(t)
                                st.rerun()

    # ─── 3D visualization panel ─────────────────────────────────────────────
    with st.container(border=True, key="viz_3d_container"):
        st.markdown(
            '<div class="section-title" style="color: #60a5fa;">Vector Space Visualization</div>'
            '<div class="section-sub">3D projection of vectors using PCA dimensionality reduction.</div>',
            unsafe_allow_html=True,
        )
        
        if expr["add"] or expr["sub"] or st.session_state.result:
            fig = create_3d_visualization(
                expr["add"], 
                expr["sub"], 
                wv, 
                saved,
                result=st.session_state.result
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="viz_3d_chart", config={'displayModeBar': True})
            else:
                st.markdown(
                    '<div style="text-align:center; padding:40px 16px; color:#565a6e;">'
                    '<div style="font-size:14px;">Add a term to see the vector</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="text-align:center; padding:40px 16px; color:#565a6e;">'
                '<div style="font-size:14px;">Add terms to see them in 3D space</div>'
                '</div>',
                unsafe_allow_html=True,
            )

with col_right:
    with st.container(border=True, key="closest_words_container"):
        st.markdown(
            '<div class="section-title">Closest Words</div>'
            '<div class="section-sub">Top semantic neighbors to the computed expression.</div>',
            unsafe_allow_html=True,
        )

        R = st.session_state.result
        if not R:
            st.markdown(
                '<div style="text-align:center; padding:40px 16px; color:#565a6e;">'
                '<div style="font-size:14px;">Awaiting expression</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="expr-box" style="margin-bottom:16px;">'
                f'<div style="font-size:11px; color:#8b8fa3; letter-spacing:0.5px; margin-bottom:4px; text-transform:uppercase; font-weight:600;">Result of</div>'
                f'<div class="expr-formula" style="color:#fbbf24; font-size:15px">{R["expr_str"]}</div></div>',
                unsafe_allow_html=True,
            )

            if R["nearest"]:
                max_sim = R["nearest"][0][1]
                for rank, (near_word, score) in enumerate(R["nearest"][:TOP_NEIGHBORS]):
                    bar = int((score / (max_sim + 1e-9)) * 100)
                    st.markdown(
                        f'<div class="neighbor-card">'
                        f'<div class="neighbor-header">'
                        f'<div><span class="neighbor-rank">{(rank + 1):02d}</span>'
                        f'<span class="neighbor-word">{near_word}</span></div>'
                        f'<span class="neighbor-score">{score:.3f}</span></div>'
                        f'<div class="neighbor-bar-bg"><div class="neighbor-bar-fill" style="width:{bar}%"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    bc1, bc2, bc3 = st.columns(3)
                    with bc1:
                        if st.button("Add", key=f"near_add_{rank}_{near_word}"):
                            if near_word not in expr["add"]:
                                expr["add"].append(near_word)
                                st.rerun()
                    with bc2:
                        if st.button("Subtract", key=f"near_sub_{rank}_{near_word}"):
                            if near_word not in expr["sub"]:
                                expr["sub"].append(near_word)
                                st.rerun()
                    with bc3:
                        if st.button("Start Fresh", key=f"near_base_{rank}_{near_word}"):
                            st.session_state.expr = {"add": [near_word], "sub": []}
                            st.session_state.result = None
                            st.rerun()

            with st.expander("Save this result"):
                save_col1, save_col2 = st.columns([3, 1])
                with save_col1:
                    custom_name = st.text_input(
                        "Name",
                        value=R["expr_str"],
                        key="save_result_name",
                        placeholder="label for this vector",
                        label_visibility="collapsed",
                    )
                with save_col2:
                    if st.button("Save", key="save_result_btn"):
                        label = custom_name.strip() or R["expr_str"]
                        vid = save_vector_from_result(label, R["vector"])
                        st.toast(f"Saved as @{vid}")
                        st.rerun()

# ─── History ───────────────────────────────────────────────────────────────
if st.session_state.history:
    with st.expander("Experiment History", expanded=True):
        with st.container(key="experiment_history"):
            for cell in st.session_state.history:
                top4 = ", ".join(w for w, _ in cell["nearest"][:TOP_NEIGHBORS])
                cell_uid = f"cell_{cell['id']}"
                st.markdown(
                    f'<div class="history-row">'
                    f'<div class="history-expr">{cell["expr_str"]}</div>'
                    f'<div class="history-res">→ {top4}</div></div>',
                    unsafe_allow_html=True,
                )
                bc1, bc2, bc3, bc4 = st.columns(4)
                with bc1:
                    if st.button("Restore", key=f"{cell_uid}_load"):
                        st.session_state.expr = {
                            "add": list(cell.get("add", [])),
                            "sub": list(cell.get("sub", [])),
                        }
                        ref = cell["vector_ref"]
                        st.session_state.result = {
                            "expr_str": cell["expr_str"],
                            "nearest": cell["nearest"],
                            "vector": saved[ref[1:]]["vector"],
                        }
                        st.rerun()
                with bc2:
                    if st.button("Add", key=f"{cell_uid}_add"):
                        ref = cell["vector_ref"]
                        if ref not in expr["add"]:
                            expr["add"].append(ref)
                            st.rerun()
                with bc3:
                    if st.button("Sub", key=f"{cell_uid}_sub"):
                        ref = cell["vector_ref"]
                        if ref not in expr["sub"]:
                            expr["sub"].append(ref)
                            st.rerun()
                with bc4:
                    if st.button("Show", key=f"{cell_uid}_show"):
                        ref = cell["vector_ref"]
                        st.session_state.result = {
                            "expr_str": cell["expr_str"],
                            "nearest": cell["nearest"],
                            "vector": saved[ref[1:]]["vector"],
                        }
                        st.rerun()

# ─── Footer ──────────────────────────────────────────────────────────────
st.markdown(
    '<p style="text-align:center; color:#ffffff; font-size:12px; font-weight:bold; margin-top:32px;">'
    "GloVe-Wiki-Gigaword-100 • Cosine Similarity • 100 Dimensions</p>",
    unsafe_allow_html=True,
)
