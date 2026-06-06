# 🧪 Embedding Lab

An interactive Streamlit app for exploring **word-vector arithmetic** with GloVe embeddings.

Type words, add or subtract them, hit **Compute**, and see which vocabulary words land closest to the resulting vector — all visualized live in a 3D projection.

---

## Run locally

**Requirements:** Python 3.11 recommended

```bash
pip install streamlit gensim==4.3.3 "numpy<2.0" "scipy<1.14" plotly scikit-learn
```

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, then click **Load Model**.

On first run, the app downloads **GloVe-Wiki-Gigaword-100** (~134 MB) via Gensim and caches it locally. Subsequent runs load straight from the cache.

---

## How it works — step by step

The app header shows a three-step flow: **Build expression → Compute → Discover.**

### Step 1 — Build Expression

Use the **Term** text field to type a vocabulary word. Or pick a previously saved result from the **Saved Vectors** dropdown (these appear after you save a Compute result).

Four buttons act on the current expression:

| Button | What it does |
|---|---|
| **＋ Add** | Appends the typed word (or selected saved vector) to the add list |
| **− Sub** | Appends it to the subtract list |
| **Clear** | Resets the entire expression and clears the last result |
| **Compute** | Evaluates the expression and finds the top 4 nearest neighbors |

Each term appears as a removable chip below the expression formula — green chips for added terms, red chips for subtracted terms. Click a chip to remove that term.

> Numbers are not accepted as terms. If a word is not in the GloVe vocabulary, a warning is shown and the term is not added.

### Step 2 — Compute

Click **Compute**. The app sums the add vectors, subtracts the subtract vectors, and queries GloVe for the 4 nearest vocabulary words by cosine similarity. Input terms are automatically excluded from results.

The expression is written in the form:

```
(paris - france) + italy
```

### Step 3 — Discover

The **Closest Words** panel (right column) shows the top 4 neighbors ranked by cosine similarity score, with a proportional bar for each. Under every neighbor are three buttons:

| Button | What it does |
|---|---|
| **Add** | Appends that neighbor to the current expression's add list |
| **Subtract** | Appends it to the subtract list |
| **Start Fresh** | Clears the expression and begins a new one with that word as the sole base term |

Expand **Save this result** to give the computed vector a name and save it. It will appear in the **Saved Vectors** dropdown as `@v1`, `@v2`, etc. and can be reused in any later expression.

---

## Vector Space Visualization

The left column also shows a **3D PCA projection** of the current expression:

- **Green arrows** — cumulative position after each added term
- **Red arrows** — cumulative position after each subtracted term
- **Origin** — fixed at (0, 0, 0)
- **After Compute** — the result vector and nearest-neighbor points are added

The 3D chart updates as you add terms, before you **Compute**. After **Compute** it switches to showing the result vector and neighbors.

> This is a PCA projection for intuition. Distances in the 3D plot do not equal distances in the original 100-dimensional space.

---

## Experiment History

Every Compute call is recorded in the **Experiment History** expander at the bottom. Each entry shows the expression and its top neighbors. Four buttons are available per entry:

| Button | What it does |
|---|---|
| **Restore** | Reloads the expression and result exactly as they were |
| **Add** | Adds the saved result vector to the current expression |
| **Sub** | Subtracts the saved result vector from the current expression |
| **Show** | Re-displays the result in the Closest Words panel without changing the expression |

---

## Example expressions to try

| Idea | Add | Subtract | Equivalent to |
|---|---|---|---|
| Country analogy | `paris`, `italy` | `france` | `paris − france + italy` |
| Historical analogy | `gandhi`, `germany` | `india` | `gandhi − india + germany` |
| Product analogy | `microsoft`, `iphone` | `apple` | `microsoft − apple + iphone` |
| Environment contrast | `arctic`, `sand` | `desert` | `arctic − desert + sand` |
| Classic gender demo | `king`, `woman` | `man` | `king − man + woman` |

Results depend on the GloVe training corpus. Some words may not be in the vocabulary.

---

## Technical details

| Item | Value |
|---|---|
| Model | `glove-wiki-gigaword-100` |
| Dimensions | 100 |
| Vocabulary | ~400 k tokens |
| Similarity metric | Cosine similarity (`KeyedVectors.similar_by_vector`) |
| Top neighbors shown | 4 (`TOP_NEIGHBORS` constant) |
| Visualization | PCA → 3 components, Plotly `Scatter3d` |
| Saved vectors | Session-only (`st.session_state`); not written to disk |

---

## Stack

- [Streamlit](https://streamlit.io) — UI
- [Gensim](https://radimrehurek.com/gensim/) — model loading and similarity search
- [GloVe](https://nlp.stanford.edu/projects/glove/) — pre-trained word vectors
- [Plotly](https://plotly.com) — 3D visualization
- [scikit-learn](https://scikit-learn.org) — PCA dimensionality reduction
- [NumPy](https://numpy.org) — vector arithmetic
