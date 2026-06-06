# 🧪 Embedding Lab

An interactive Streamlit app for exploring **word-vector arithmetic** using GloVe embeddings.

Instead of treating embeddings as abstract math, Embedding Lab lets you build additive/subtractive expressions over real word vectors, inspect semantic neighbors by cosine similarity, and watch meaning move through a 3D PCA-projected space — all in a live UI.

---

## What it does

Word embeddings encode semantic relationships as directions in high-dimensional space. Embedding Lab makes that tangible:

- Type a word, add or subtract it from an expression
- Hit **Compute** to evaluate the vector sum and find the nearest vocabulary words
- Watch the 3D projection update in real time as you build your expression
- Save computed vectors and reuse them as inputs to later expressions
- Browse session history to restore or chain earlier results

---

## Quick start

**Requirements:** Python 3.11 recommended

```bash
pip install streamlit gensim==4.3.3 "numpy<2.0" "scipy<1.14" plotly scikit-learn
```

```bash
streamlit run embeddings_intuition.py
```

1. Open the local URL printed in the terminal.
2. Click **Load Model** — the app downloads **GloVe-Wiki-Gigaword-100** (~134 MB) once and caches it locally.
3. Build an expression and click **Compute**.

---

## How to use it

### 1 — Build an expression

Enter any vocabulary word and click **Add** (green) or **Subtract** (red). Each term appears as a removable chip in the expression panel. You can also add a previously saved result by typing its reference, e.g. `@v1`.

### 2 — Compute

Click **Compute** to evaluate the expression vector and retrieve the top 4 nearest neighbors by cosine similarity. The result formula is displayed in the Closest Words panel.

### 3 — Explore neighbors

Each neighbor card shows the word, its similarity score, and a proportional bar. From any neighbor you can:

| Action | What it does |
|---|---|
| **Add** | Appends the word to the current expression's add list |
| **Subtract** | Appends the word to the current expression's subtract list |
| **Start Fresh** | Clears the expression and starts a new one with that word as the base |

### 4 — Save a result

Expand **Save this result** below the neighbor cards, optionally rename the vector, and click **Save**. The vector is stored as `@v1`, `@v2`, etc. and becomes available for use in any future expression.

### 5 — Visualize in 3D

The **Vector Space Visualization** panel renders a Plotly 3D scatter/arrow chart:

- **Green arrows** — added terms
- **Red arrows** — subtracted terms
- **Origin** — anchored at (0, 0, 0)
- **Result + neighbors** — shown after Compute

Vectors are projected to three dimensions with PCA. This is an intuition aid; spatial distances in the plot do not correspond to distances in the original 100-dimensional space.

### 6 — Experiment History

All past Compute calls are logged. For each entry:

| Button | Effect |
|---|---|
| **Restore** | Reloads the expression and result exactly as they were |
| **Add** | Adds the saved result vector to the current expression |
| **Sub** | Subtracts the saved result vector from the current expression |
| **Show** | Re-displays the result in the Closest Words panel without changing the expression |

---

## Example expressions

| Concept | Add | Subtract | Classic framing |
|---|---|---|---|
| Country analogy | `paris`, `italy` | `france` | `paris − france + italy` |
| Historical analogy | `gandhi`, `germany` | `india` | `gandhi − india + germany` |
| Product analogy | `microsoft`, `iphone` | `apple` | `microsoft − apple + iphone` |
| Environment contrast | `arctic`, `sand` | `desert` | `arctic − desert + sand` |
| Gender analogy | `king`, `woman` | `man` | `king − man + woman` |

Results depend on the GloVe training corpus. Some words may not be in the vocabulary.

---

## Technical details

| Component | Detail |
|---|---|
| **Model** | `glove-wiki-gigaword-100` via Gensim Downloader |
| **Dimensions** | 100 |
| **Vocabulary** | ~400k tokens |
| **Similarity** | Cosine similarity (`KeyedVectors.similar_by_vector`) |
| **Visualization** | PCA → 3 components, Plotly `Scatter3d` |
| **Top neighbors** | 4 (configurable via `TOP_NEIGHBORS` constant) |

---

## Stack

- [Streamlit](https://streamlit.io) — UI framework
- [Gensim](https://radimrehurek.com/gensim/) — model loading and similarity search
- [GloVe](https://nlp.stanford.edu/projects/glove/) — pre-trained word vectors
- [Plotly](https://plotly.com) — interactive 3D visualization
- [scikit-learn](https://scikit-learn.org) — PCA dimensionality reduction
- [NumPy](https://numpy.org) — vector arithmetic

---

## Notes

- Only single-word tokens present in the GloVe vocabulary are accepted; numbers and phrases will not resolve.
- Nearest-neighbor search excludes the input terms themselves from results.
- The PCA projection is recomputed each time the expression changes, so axis scales shift as terms are added.
- Saved vectors (`@v1`, `@v2`, …) persist only for the current browser session; they are not written to disk.
