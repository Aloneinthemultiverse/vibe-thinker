"""Embedding source for the similarity index.

GitNexus can generate embeddings but its similarity SEARCH doesn't scale on this
platform (see vectorstore.py). So we take only the embedding *vectors* and put them
in our own TurboQuant index. Here we use a small cached SentenceTransformer
(all-MiniLM-L6-v2, 384-dim) — already on disk, no download. dim=384 matches the
TurboQuantIndex default.
"""
import os

# Use the local HF cache only — never block on the network at inference time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

_MODEL = None
_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_NAME)
    return _MODEL


def embed(texts):
    """Return an (n, DIM) float64 array of L2-normalized embeddings."""
    if isinstance(texts, str):
        texts = [texts]
    v = _model().encode(list(texts), normalize_embeddings=True,
                        show_progress_bar=False)
    return np.asarray(v, dtype=np.float64)
