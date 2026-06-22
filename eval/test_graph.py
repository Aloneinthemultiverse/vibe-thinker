"""Phase B test: the typed blackboard graph (agent/graph.py).

Run: python -m eval.test_graph   (HashEmbedder, temp store — no servers needed)
"""
import os
import shutil
import tempfile

os.environ.setdefault("EMBED_DIM", "512")  # ensure hash embedder
from agent.graph import Graph, KINDS, EDGE_TYPES
from agent.vecindex import HashEmbedder


def _g():
    tmp = tempfile.mkdtemp(prefix="graph_test_")
    return Graph(tmp, embedder=HashEmbedder(), name="t"), tmp


def test_add_and_get():
    g, tmp = _g()
    try:
        nid = g.add_node("insight" if "insight" in KINDS else "directive", "reasoner",
                         "initials joins first letter of each word uppercased")
        assert g.get(nid)["author"] == "reasoner"
        assert g.vi.ntotal == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collision_free_ids():
    """§9.4 — identical text added twice must get DIFFERENT ids (seq differs)."""
    g, tmp = _g()
    try:
        a = g.add_node("result", "actor", "tests still fail")
        b = g.add_node("result", "actor", "tests still fail")
        assert a != b, "identical text collided -> overwrite bug"
        assert len(g.nodes) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_self_wiring():
    """Nodes sharing an identifier auto-link with an 'about' edge, no LLM."""
    g, tmp = _g()
    try:
        a = g.add_node("directive", "reasoner", "fix the initials function casing")
        b = g.add_node("result", "actor", "initials returned lowercase, still failing")
        assert b in g.neighbors(a) or a in g.neighbors(b), "shared symbol did not wire"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_explicit_link_and_neighbors():
    g, tmp = _g()
    try:
        a = g.add_node("directive", "reasoner", "alpha plan", wire=False)
        b = g.add_node("result", "actor", "beta outcome", wire=False)
        g.link(a, b, "led_to")
        assert b in g.neighbors(a) and a in g.neighbors(b)
        assert b in g.neighbors(a, type="led_to")
        assert g.neighbors(a, type="refuted") == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_and_think():
    g, tmp = _g()
    try:
        g.add_node("fact", "system", "python str.title uppercases after every non letter")
        g.add_node("directive", "reasoner", "initials uppercase join first letters")
        g.add_node("result", "actor", "reverse list works fine")
        hits = g.search("initials uppercase", k=2)
        assert hits and "initials" in hits[0]["text"]
        t = g.think("initials uppercase")
        assert t["answer"] and "gap" in t
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_recent_and_window():
    g, tmp = _g()
    try:
        for i in range(6):
            g.add_node("action", "actor", f"step number {i} editing buggy file")
        r = g.recent(n=3)
        assert len(r) == 3 and r[0]["text"].endswith("5 editing buggy file")
        w = g.window("buggy file", recent_k=2, search_m=2, max_chars=2500)
        assert w and len(w) <= 2500
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persistence():
    g, tmp = _g()
    try:
        g.add_node("plan", "reasoner", "decompose into two subgoals foo and bar")
        g.add_node("result", "actor", "foo subgoal complete")
        seq_before = g._seq
        g2 = Graph(tmp, embedder=HashEmbedder(), name="t")
        assert len(g2.nodes) == 2 and g2._seq == seq_before
        assert g2.vi.ntotal == 2
        assert g2.search("foo subgoal", k=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_kind_and_edge_rejected():
    g, tmp = _g()
    try:
        for bad in ("nonsense", ""):
            try:
                g.add_node(bad, "actor", "x"); assert False, "bad kind accepted"
            except ValueError:
                pass
        a = g.add_node("fact", "system", "x")
        try:
            g.link(a, a, "bogus"); assert False, "bad edge accepted"
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_source_tier_boost():
    """gbrain source-tier: a code 'fact' node must outrank session chatter for the same
    query, so chatter can't crowd out the real content (the duo2 retrieval bug)."""
    g, tmp = _g()
    try:
        # lots of chatter mentioning 'initials', plus ONE real code fact
        for i in range(5):
            g.add_node("result", "actor", f"attempt {i}: initials still failing on case")
            g.add_node("directive", "reasoner", f"round {i}: fix initials casing somehow")
        g.add_node("fact", "system",
                   "FILE buggy.py:\ndef initials(name):\n    return ''.join(w[0] for w in name.split()).upper()")
        hits = g.search("initials function code", k=3)
        assert hits[0]["kind"] == "fact", f"chatter crowded out code: top={hits[0]['kind']}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_retrieve_content_clean():
    """retrieve_content returns CLEAN source text (no [kind/author] labels, no chatter)."""
    g, tmp = _g()
    try:
        g.add_node("directive", "reasoner", "noise about initials we should ignore")
        g.add_node("fact", "system", "def initials(name):\n    return 'AL'  # the code")
        content = g.retrieve_content("initials code", k=2)
        assert "def initials" in content
        assert "[directive" not in content and "noise about" not in content
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
