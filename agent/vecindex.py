"""Reusable hybrid-retrieval index: embedder + FAISS (flat -> IVF-PQ) + BM25 + RRF.

Extracted from episodic.py so both episodic.py (the brain) and graph.py (the dual-brain
blackboard) share ONE implementation of vector + keyword retrieval. Behaviour is identical
to the original episodic internals — this is a pure refactor.

VecIndex deliberately knows nothing about tasks/solutions/nodes. It indexes opaque
documents: you give it the text to embed and the text to BM25, it gives back rankings.
The caller (EpisodicMemory, Graph) owns the payloads and any extra signal (e.g. graph
adjacency) and fuses everything via `rrf`.
"""
import json
import math
import os
import re

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - faiss optional at import time
    faiss = None

DIM = int(os.environ.get("EMBED_DIM", "512"))
IVFPQ_THRESHOLD = int(os.environ.get("EPISODIC_IVFPQ_AFTER", "2000"))

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[(){}\[\].:=+\-*/%<>!]+")


def _tok(text):
    return _TOKEN.findall((text or "").lower())


# --------------------------------------------------------------------------- embedders
class HashEmbedder:
    """Deterministic hashing vectorizer over code/text tokens. No model, no GPU,
    fully on-device. Good enough to recall structurally-similar tasks; swap in a
    neural embedder for semantic recall once an embed server is available."""

    dim = DIM

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall((t or "").lower()):
                h = hash(tok)
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        # L2-normalize so inner product == cosine similarity.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class LlamaEmbedder:
    """Neural embeddings from a local OpenAI-compatible /v1/embeddings endpoint
    (e.g. a second llama-server running a small embed GGUF). Set EMBED_ENDPOINT."""

    def __init__(self, endpoint=None):
        self.endpoint = endpoint or os.environ.get("EMBED_ENDPOINT")
        self._dim = None

    @property
    def dim(self):
        return self._dim or DIM

    def embed(self, texts):
        import urllib.request
        body = json.dumps({"input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        vecs = np.array([d["embedding"] for d in data["data"]], dtype="float32")
        self._dim = vecs.shape[1]
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return vecs / n


def default_embedder():
    """LlamaEmbedder if an embed server is configured, else the on-device hash one."""
    if os.environ.get("EMBED_ENDPOINT"):
        return LlamaEmbedder()
    return HashEmbedder()


# --------------------------------------------------------------------------------- bm25
class BM25:
    """Tiny in-memory BM25 keyword scorer (Okapi). No deps; rebuilt from payloads
    on load. Catches exact-token matches that dense vectors miss (e.g. a function
    name), the keyword half of hybrid retrieval."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs, self.df, self.avgdl = [], {}, 0.0

    def add(self, tokens):
        self.docs.append(tokens)
        for t in set(tokens):
            self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)

    def scores(self, q_tokens):
        n = len(self.docs)
        out = [0.0] * n
        if not n:
            return out
        q = set(q_tokens)
        for i, doc in enumerate(self.docs):
            dl = len(doc) or 1
            tf = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (tf[t] * (self.k1 + 1)) / denom
            out[i] = s
        return out


def rrf(rankings, k=60):
    """Reciprocal-rank fusion: combine several ranked index-lists into one score
    dict. Robust to incomparable score scales (cosine vs BM25 vs graph overlap)."""
    fused = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


# ------------------------------------------------------------------------------- index
class VecIndex:
    """FAISS vector index (exact flat, auto-upgrading to quantized IVF-PQ at scale)
    paired with a BM25 keyword index over the same documents. Index-position `i`
    is shared across both and with the caller's payload list."""

    def __init__(self, embedder=None, dim=None):
        if faiss is None:
            raise RuntimeError("faiss not installed; pip install faiss-cpu")
        self.embedder = embedder or default_embedder()
        self.dim = dim or self.embedder.dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.bm25 = BM25()

    @property
    def ntotal(self):
        return self.index.ntotal

    # --- persistence (vector index only; BM25 is rebuilt from caller payloads) ---
    def load(self, path):
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            self.dim = self.index.d

    def save(self, path):
        faiss.write_index(self.index, path)

    def rebuild_bm25(self, texts):
        self.bm25 = BM25()
        for t in texts:
            self.bm25.add(_tok(t))

    # --- write path ---
    def add(self, embed_text, bm25_text=None):
        vec = self.embedder.embed([embed_text])
        if vec.shape[1] != self.index.d:  # embedder changed dim -> rebuild flat
            self.index = faiss.IndexFlatIP(vec.shape[1])
            self.dim = vec.shape[1]
        self.index.add(vec)
        self.bm25.add(_tok(bm25_text if bm25_text is not None else embed_text))
        self._maybe_quantize()

    def _maybe_quantize(self):
        """Past a threshold, swap the exact flat index for a QUANTIZED IVF-PQ one
        (trained on the accumulated vectors) for fast, memory-cheap search at scale."""
        n = self.index.ntotal
        if n < IVFPQ_THRESHOLD or isinstance(self.index, faiss.IndexIVFPQ):
            return
        vecs = self.index.reconstruct_n(0, n) if hasattr(self.index, "reconstruct_n") else None
        if vecs is None:
            return
        nlist = max(1, int(np.sqrt(n)))
        m = 8 if self.dim % 8 == 0 else 4   # PQ subquantizers must divide dim
        quant = faiss.IndexFlatIP(self.dim)
        ivf = faiss.IndexIVFPQ(quant, self.dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        ivf.train(vecs)
        ivf.add(vecs)
        ivf.nprobe = min(nlist, 8)
        self.index = ivf

    # --- read path ---
    def vector_rank(self, query_text, limit=None):
        """Return (ranked index list, {idx: cosine}) for the query over all rows."""
        n = self.index.ntotal
        if n == 0:
            return [], {}
        k = n if limit is None else min(limit, n)
        vec = self.embedder.embed([query_text])
        scores, idx = self.index.search(vec, k)
        rank = [int(i) for i in idx[0] if i >= 0]
        score = {int(i): float(s) for i, s in zip(idx[0], scores[0]) if i >= 0}
        return rank, score

    def bm25_scores(self, query_text):
        return self.bm25.scores(_tok(query_text))

    def bm25_rank(self, query_text):
        bscores = self.bm25_scores(query_text)
        rank = [i for i in sorted(range(len(bscores)), key=lambda j: bscores[j],
                                  reverse=True) if bscores[i] > 0]
        return rank, bscores
