"""Episodic ('brain') memory: retrieve worked examples to put in the model's context.

The thesis (harness = intelligence) extended: instead of hoping the 3B *invents* a
solution from cold, the harness RECALLS similar solved tasks and injects them into the
prompt, so the small model ADAPTS a near-example rather than reasoning from scratch.

Two pieces, both swappable:
  - Embedder: text -> vector. Defaults to a zero-dependency deterministic HashEmbedder
    (works on-device with no model). Plug in LlamaEmbedder (a local embed server) for
    real neural embeddings by setting EMBED_ENDPOINT — no other code changes.
  - Index: FAISS. Exact inner-product (IndexFlatIP) at small scale; auto-upgrades to a
    QUANTIZED IVF-PQ index past a threshold (the 'turboquant' fast-search angle) so it
    stays cheap as the corpus of solved tasks grows.

Persists to runtime/episodic/ as a FAISS index + a JSONL payload sidecar.
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
STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "runtime", "episodic")
IVFPQ_THRESHOLD = int(os.environ.get("EPISODIC_IVFPQ_AFTER", "2000"))

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[(){}\[\].:=+\-*/%<>!]+")


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
def _tok(text):
    return _TOKEN.findall((text or "").lower())


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


def _rrf(rankings, k=60):
    """Reciprocal-rank fusion: combine several ranked index-lists into one score
    dict. Robust to incomparable score scales (cosine vs BM25 vs graph overlap)."""
    fused = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


# ------------------------------------------------------------------------------- store
class EpisodicMemory:
    """A searchable log of (task -> solution) episodes the agent can recall."""

    def __init__(self, embedder=None, store_dir=STORE_DIR):
        if faiss is None:
            raise RuntimeError("faiss not installed; pip install faiss-cpu")
        self.embedder = embedder or default_embedder()
        self.dir = os.path.abspath(store_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.idx_path = os.path.join(self.dir, "index.faiss")
        self.pay_path = os.path.join(self.dir, "payloads.jsonl")
        self.payloads = []
        self.dim = self.embedder.dim
        self.index = None
        self._load()

    # --- persistence ---
    def _load(self):
        if os.path.exists(self.pay_path):
            with open(self.pay_path, "r", encoding="utf-8") as f:
                self.payloads = [json.loads(l) for l in f if l.strip()]
        if os.path.exists(self.idx_path):
            self.index = faiss.read_index(self.idx_path)
            self.dim = self.index.d
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        # BM25 is cheap to rebuild from payloads — no separate persistence needed.
        self.bm25 = BM25()
        for p in self.payloads:
            self.bm25.add(_tok(p["task"] + " " + p["solution"]))

    def _save(self):
        faiss.write_index(self.index, self.idx_path)
        with open(self.pay_path, "w", encoding="utf-8") as f:
            for p in self.payloads:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- write path ---
    def add(self, task, solution, meta=None):
        """Record a solved episode. `solution` is the corrected file/answer text."""
        vec = self.embedder.embed([task])
        if vec.shape[1] != self.index.d:  # embedder changed dim -> rebuild flat
            self.index = faiss.IndexFlatIP(vec.shape[1])
            self.dim = vec.shape[1]
        self.index.add(vec)
        self.payloads.append({"task": task, "solution": solution, "meta": meta or {}})
        self.bm25.add(_tok(task + " " + solution))
        self._maybe_quantize()
        self._save()

    def _graph_rank(self, task):
        """GitNexus graph signal: find symbols related to the task, rank episodes by
        how many of those symbols they mention. Degrades to [] when the concept isn't
        in an indexed repo (e.g. toy sandboxes) — honest no-op, never an error."""
        try:
            from .retriever import default_retriever
            res = default_retriever().query(task)
            if not isinstance(res, str) or res.startswith("ERROR"):
                return []
            syms = set(_tok(res))
            scored = []
            for i, p in enumerate(self.payloads):
                text = set(_tok(p["task"] + " " + p["solution"]))
                overlap = len(syms & text)
                if overlap:
                    scored.append((overlap, i))
            scored.sort(reverse=True)
            return [i for _, i in scored]
        except Exception:
            return []

    def _maybe_quantize(self):
        """Past a threshold, swap the exact flat index for a QUANTIZED IVF-PQ one
        (trained on the accumulated vectors) for fast, memory-cheap search at scale."""
        n = len(self.payloads)
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
    def search(self, task, k=2, use_graph=True):
        """HYBRID retrieval (the gbrain recipe): fuse dense-vector + BM25-keyword +
        GitNexus-graph rankings with reciprocal-rank fusion. Returns up to k episodes
        [{task, solution, score, vec, bm25}] best-first. Fusion sidesteps the problem
        that the three signals have incomparable score scales."""
        n = self.index.ntotal
        if n == 0:
            return []

        # 1) dense vector ranking
        vec = self.embedder.embed([task])
        vscores, vidx = self.index.search(vec, n)
        vec_rank = [int(i) for i in vidx[0] if 0 <= i < len(self.payloads)]
        vec_score = {int(i): float(s) for i, s in zip(vidx[0], vscores[0]) if i >= 0}

        # 2) BM25 keyword ranking
        bscores = self.bm25.scores(_tok(task))
        bm_rank = [i for i in sorted(range(len(bscores)), key=lambda j: bscores[j],
                                     reverse=True) if bscores[i] > 0]

        # 3) graph ranking (no-op off-repo)
        g_rank = self._graph_rank(task) if use_graph else []

        fused = _rrf([r for r in (vec_rank, bm_rank, g_rank) if r])
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]

        hits = []
        for i in order:
            p = self.payloads[i]
            hits.append({"task": p["task"], "solution": p["solution"],
                         "score": round(fused[i], 4),
                         "vec": round(vec_score.get(i, 0.0), 3),
                         "bm25": round(bscores[i] if i < len(bscores) else 0.0, 2),
                         "graph": i in g_rank})
        return hits


_RET_RE = re.compile(r"^\s*(def\s+\w+\([^)]*\)|return\b.*|\w+\s*=\s*.+)$")


def _core_lines(solution, limit=3):
    """Pull the signature + the key return/assignment lines from a solution — the
    idiom worth recalling, not the whole file."""
    keep = []
    for ln in solution.splitlines():
        if _RET_RE.match(ln) and "def __" not in ln:
            keep.append(ln.strip())
        if len(keep) >= limit:
            break
    return keep


def synthesize_recall(hits, max_chars=700):
    """think-style recall (gbrain `think`, not `search`): instead of pasting raw
    example files, synthesize ONE focused guidance block — the shared approach, the
    key idiom(s), and an explicit GAP/what-to-watch note. Keeps the small model's
    context tight and tells it what it still has to get right itself."""
    if not hits:
        return ""
    lines = ["BRAIN RECALL (synthesized from your past solved work — adapt, don't copy):"]
    idioms = []
    for h in hits:
        for c in _core_lines(h["solution"]):
            if c not in idioms:
                idioms.append(c)
    lines.append("Approach that worked on similar tasks:")
    for c in idioms[:5]:
        lines.append(f"  - {c}")
    top = hits[0]
    lines.append(f"Closest prior task: {top['task'][:140]}")
    lines.append("GAP — still YOUR job: make the output match the test's EXACT expected "
                 "values (types, order, case). Each bug is independent; fix every one.")
    block = "\n".join(lines)
    return block[:max_chars]


def render_examples(hits, max_chars=1200):
    """Raw-paste recall (gbrain `search` style). Kept for debugging / comparison;
    the loop uses synthesize_recall by default."""
    if not hits:
        return ""
    parts = ["RELEVANT SOLVED EXAMPLES (adapt these — do not copy blindly):"]
    for h in hits:
        sol = h["solution"].strip()
        if len(sol) > max_chars:
            sol = sol[:max_chars] + "\n# ...(truncated)"
        parts.append(f"\n# similar task: {h['task'][:160]}\n```python\n{sol}\n```")
    return "\n".join(parts)
