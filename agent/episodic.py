"""Episodic ('brain') memory: retrieve worked examples to put in the model's context.

The thesis (harness = intelligence) extended: instead of hoping the 3B *invents* a
solution from cold, the harness RECALLS similar solved tasks and injects them into the
prompt, so the small model ADAPTS a near-example rather than reasoning from scratch.

The retrieval machinery (embedder + FAISS flat/IVF-PQ + BM25 + RRF) lives in vecindex.py
and is shared with graph.py. This module adds the episode payloads, the GitNexus graph
signal, and the think-style recall synthesis on top.

Persists to runtime/episodic/ as a FAISS index + a JSONL payload sidecar.
"""
import json
import os
import re

# Re-exported for backward compatibility (older imports expect these here).
from .vecindex import (  # noqa: F401
    DIM, IVFPQ_THRESHOLD, BM25, HashEmbedder, LlamaEmbedder, VecIndex,
    default_embedder, rrf as _rrf, _tok,
)

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "runtime", "episodic")


# ------------------------------------------------------------------------------- store
class EpisodicMemory:
    """A searchable log of (task -> solution) episodes the agent can recall."""

    def __init__(self, embedder=None, store_dir=STORE_DIR):
        self.dir = os.path.abspath(store_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.idx_path = os.path.join(self.dir, "index.faiss")
        self.pay_path = os.path.join(self.dir, "payloads.jsonl")
        self.payloads = []
        self.vi = VecIndex(embedder=embedder)
        self.embedder = self.vi.embedder
        self._load()

    @property
    def dim(self):
        return self.vi.dim

    @property
    def index(self):  # kept for callers/tests that peek at the raw faiss index
        return self.vi.index

    # --- persistence ---
    def _load(self):
        if os.path.exists(self.pay_path):
            with open(self.pay_path, "r", encoding="utf-8") as f:
                self.payloads = [json.loads(l) for l in f if l.strip()]
        self.vi.load(self.idx_path)
        # BM25 is cheap to rebuild from payloads — no separate persistence needed.
        self.vi.rebuild_bm25([p["task"] + " " + p["solution"] for p in self.payloads])

    def _save(self):
        self.vi.save(self.idx_path)
        with open(self.pay_path, "w", encoding="utf-8") as f:
            for p in self.payloads:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- write path ---
    def add(self, task, solution, meta=None):
        """Record a solved episode. `solution` is the corrected file/answer text."""
        self.vi.add(task, bm25_text=task + " " + solution)
        self.payloads.append({"task": task, "solution": solution, "meta": meta or {}})
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

    # --- read path ---
    def search(self, task, k=2, use_graph=True):
        """HYBRID retrieval (the gbrain recipe): fuse dense-vector + BM25-keyword +
        GitNexus-graph rankings with reciprocal-rank fusion. Returns up to k episodes
        [{task, solution, score, vec, bm25}] best-first. Fusion sidesteps the problem
        that the three signals have incomparable score scales."""
        if self.vi.ntotal == 0:
            return []

        vec_rank, vec_score = self.vi.vector_rank(task)
        vec_rank = [i for i in vec_rank if i < len(self.payloads)]
        bm_rank, bscores = self.vi.bm25_rank(task)
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
