"""Phase A test: prove the vecindex.py refactor did not change episodic behaviour.

Run: python -m eval.test_vecindex
Uses HashEmbedder (no embed server needed) and a throwaway temp store.
"""
import os
import shutil
import tempfile

from agent.episodic import (
    EpisodicMemory, synthesize_recall, render_examples,
    default_embedder, HashEmbedder, _rrf,
)
from agent.vecindex import VecIndex, BM25, rrf


def _fresh_mem():
    tmp = tempfile.mkdtemp(prefix="epi_test_")
    return EpisodicMemory(embedder=HashEmbedder(), store_dir=tmp), tmp


def test_backward_compat_imports():
    # names that other modules import from episodic must still resolve
    assert callable(default_embedder)
    assert _rrf is rrf
    assert isinstance(HashEmbedder(), HashEmbedder)


def test_rrf_fuses():
    fused = rrf([[2, 0, 1], [0, 2, 1]])
    # idx 0 and 2 each top one ranking -> beat idx 1 which is mid in both
    assert fused[0] > fused[1] and fused[2] > fused[1]


def test_add_search_recall():
    mem, tmp = _fresh_mem()
    try:
        mem.add("fix to_celsius conversion bug add plus one",
                "def to_celsius(f):\n    return (f - 32) * 5 / 9")
        mem.add("compute initials uppercase from full name",
                "def initials(s):\n    return ''.join(w[0] for w in s.split()).upper()")
        mem.add("reverse a list of integers",
                "def rev(xs):\n    return xs[::-1]")
        assert mem.index.ntotal == 3

        hits = mem.search("initials uppercase name", k=2, use_graph=False)
        assert hits, "search returned no hits"
        # most relevant episode should surface first
        assert "initials" in hits[0]["task"]
        for h in hits:
            assert {"task", "solution", "score", "vec", "bm25", "graph"} <= set(h)

        block = synthesize_recall(hits)
        assert "BRAIN RECALL" in block and "GAP" in block
        assert render_examples(hits)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persistence_roundtrip():
    mem, tmp = _fresh_mem()
    try:
        mem.add("dedupe a list keep order", "def dd(xs):\n    return list(dict.fromkeys(xs))")
        # reopen from disk -> payload + index survive, search still works
        mem2 = EpisodicMemory(embedder=HashEmbedder(), store_dir=tmp)
        assert len(mem2.payloads) == 1
        assert mem2.index.ntotal == 1
        assert mem2.search("dedupe order", k=1, use_graph=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_vecindex_standalone():
    vi = VecIndex(embedder=HashEmbedder())
    vi.add("alpha beta gamma", "alpha beta gamma")
    vi.add("delta epsilon", "delta epsilon")
    assert vi.ntotal == 2
    rank, score = vi.vector_rank("alpha beta")
    assert rank and rank[0] == 0
    brank, bscores = vi.bm25_rank("delta")
    assert brank and brank[0] == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
