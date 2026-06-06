# Embedding Lab

Interactive education material for the concept of "embeddings" in any GenAI Program.
This app is designed to demystify word embeddings for learners who are new to AI. Instead of treating embeddings as abstract mathematical objects, the lab turns them into something participants can experiment with, test, and visually explore.

## Why this lab is useful

For many beginners, word embeddings can feel confusing: words become numbers, meanings become vectors, and relationships between ideas are hidden inside high-dimensional space.

This lab makes the concept more intuitive.

Participants can build simple semantic expressions, such as `paris - france + italy`, and immediately see what the model considers semantically close to the result. They can also visualize the movement of these word vectors in a simplified 3D space, helping them understand that embeddings are not just stored numbers, but directions and relationships in meaning-space.

Instead of only explaining embeddings theoretically, this lab lets participants learn by doing.

They can:
- add and subtract words to form vector expressions,
- test analogies and semantic relationships,
- discover nearest semantic neighbors to a computed vector,
- reuse previous results as saved vectors,
- see a 3D PCA projection of vector movement,
- compare how different words shift meaning in semantic space.

The goal is to help new learners move from “embeddings are mysterious numbers” to “embeddings capture meaningful relationships between words.”

## How to use

1. Open the Streamlit app link.
2. Click **Load Model**.
3. Try expressions such as:
   - `gandhi - india + germany`
   - `paris - france + italy`
   - `doctor - hospital + school`

## Built with

- Streamlit
- Gensim
- GloVe embeddings
- Plotly
- scikit-learn
