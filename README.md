# Embedding Lab

Interactive learning tool for understanding **word embeddings** in generative AI and NLP.

Instead of treating embeddings as abstract math, this lab lets learners build vector expressions, inspect semantic neighbors, and explore how meaning moves in space. The goal is to move from *"embeddings are mysterious numbers"* to *"embeddings encode relationships between words."*

## Why this lab is useful

For many beginners, embeddings feel hard to grasp: words become vectors, meaning lives in high-dimensional space, and relationships are hidden behind linear algebra.

Embedding Lab makes the idea concrete. Participants can:

- build expressions by adding and subtracting words,
- compute a resulting semantic vector,
- inspect the closest words by cosine similarity,
- visualize vector steps in a 3D PCA projection,
- save results and reuse them in later expressions,
- review past experiments from session history.

The app uses **GloVe-Wiki-Gigaword-100** (100-dimensional word vectors). On first run, the model downloads once (~134 MB) and is cached locally afterward.

## What you can do

### 1. Build an expression

Enter a vocabulary term and click **Add** or **Subtract**. Terms appear as removable chips in the expression panel.

You can also reuse saved vectors from earlier results, for example `@v1`.

### 2. Compute

Click **Compute** to evaluate the expression and find the top semantic neighbors.

Example idea:

- add `paris`
- subtract `france`
- add `italy`

This is equivalent to exploring:

```text
paris - france + italy
```

### 3. Discover

The **Closest Words** panel shows the top 4 neighbors with similarity scores.

From there you can:

- Add or Subtract a neighbor into a new expression,
- Start Fresh with a neighbor as a new base term,
- Save the computed result for reuse.

### 4. Visualize

The **Vector Space Visualization** panel shows a 3D PCA projection of the expression:

- green arrows for added terms,
- red arrows for subtracted terms,
- the origin anchored at (0, 0, 0),
- after compute: the result vector and nearest neighbors.

This is a simplified view for intuition, not the full 100-dimensional embedding space.

### 5. Revisit experiments

**Experiment History** stores past computations.

For each entry you can:

- Restore the expression and result,
- Add or Subtract the saved vector into the current expression,
- Show the result again in the results panel.

## Example expressions to try

| Concept | Terms to add/subtract |
|---|---|
| Country analogy | paris − france + italy |
| Historical analogy | gandhi − india + germany |
| Company/product shift | microsoft − apple + iphone |
| Environment contrast | arctic − desert + sand |
| Classic embedding demo | king − man + woman |

Results depend on the GloVe vocabulary and training data. Some terms may not exist in the model.

## How to run locally

**Requirements:** Python 3.11 recommended

Install dependencies:

```bash
pip install streamlit gensim==4.3.3 "numpy<2.0" "scipy<1.14" plotly scikit-learn
```

Start the app:

```bash
streamlit run app.py
```

Then:

1. Open the local URL shown in the terminal.
2. Click **Load Model**.
3. Build an expression and click **Compute**.

## Built with

- [Streamlit](https://streamlit.io) — interactive UI
- [Gensim](https://radimrehurek.com/gensim/) — embedding model loading and similarity search
- [GloVe](https://nlp.stanford.edu/projects/glove/) — pre-trained word vectors (glove-wiki-gigaword-100)
- [Plotly](https://plotly.com) — 3D vector visualization
- [scikit-learn](https://scikit-learn.org) — PCA dimensionality reduction

## Notes for learners

- Only single-word terms in the model vocabulary are supported.
- Numbers are not accepted as terms.
- Nearest-neighbor results are based on cosine similarity.
- The 3D plot is a PCA projection for visualization; distances in the plot are not the same as distances in the original 100D space.
