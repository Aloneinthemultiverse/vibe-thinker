"""Lightweight typed knowledge graph — the dual-brain blackboard (and per-brain stores).

Built on vecindex.VecIndex (shared vector + BM25 machinery). Borrows from gbrain:
  - self-wiring typed edges: on every write, link the new node to recent nodes that
    share a symbol/identifier — ZERO LLM calls (pure token overlap).
  - hybrid retrieval: vector + BM25 + graph-adjacency fused with reciprocal-rank fusion.
  - think vs search: search() returns raw nodes; think() returns a synthesized answer
    plus a GAP note (what the graph doesn't know yet).

Design notes wired in here:
  - §9.4 collision-free IDs: id = sha1(author|kind|seq|text); seq is a persisted
    monotonic counter, so identical text at different times gets distinct ids. No
    clock / RNG -> resume-safe and deterministic.
  - §9.5 bounded context: window() returns a size-capped, role-shaped feed (recent
    tail + top search hits) so a brain never sees the whole graph.

Persists to <store_dir>/{nodes.jsonl, edges.jsonl, index.faiss, meta.json}.
"""
import hashlib
import json
import os

from .vecindex import VecIndex, _tok, rrf, default_embedder

KINDS = ("plan", "directive", "correction", "query", "action", "result", "fact", "symbol")
EDGE_TYPES = ("led_to", "refuted", "supports", "about", "calls")

# gbrain-style SOURCE TIERS: authoritative content (code, docs, symbols) outranks
# transient session chatter (directives/results/actions) in retrieval, so chatter never
# crowds out the real code (the duo2 bug). Higher = more authoritative.
TIER = {"fact": 1.6, "symbol": 1.6, "plan": 1.1, "directive": 0.8, "correction": 0.8,
        "query": 0.7, "action": 0.7, "result": 0.7}
SOURCE_KINDS = ("fact", "symbol")   # clean content lives here (code/docs ingested via markitdown etc.)

# tokens too generic to be worth wiring an edge on
_STOP = {"the", "a", "an", "is", "to", "of", "and", "or", "in", "on", "for", "it",
         "def", "return", "self", "if", "else", "not", "true", "false", "none"}


def _ident_tokens(text):
    """Identifier-like tokens worth linking on (drop punctuation + stopwords)."""
    out = []
    for t in _tok(text):
        if t.isidentifier() and t not in _STOP and len(t) > 2:
            out.append(t)
    return out


class Graph:
    def __init__(self, store_dir, embedder=None, name="graph"):
        self.name = name
        self.dir = os.path.abspath(store_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.nodes_path = os.path.join(self.dir, "nodes.jsonl")
        self.edges_path = os.path.join(self.dir, "edges.jsonl")
        self.idx_path = os.path.join(self.dir, "index.faiss")
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.nodes = []            # list of node dicts; position i == VecIndex row i
        self.edges = []            # list of edge dicts
        self.id2pos = {}
        self._seq = 0
        self.vi = VecIndex(embedder=embedder or default_embedder())
        self._load()

    # ----------------------------------------------------------------- persistence
    def _load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._seq = json.load(f).get("seq", 0)
        if os.path.exists(self.nodes_path):
            with open(self.nodes_path, "r", encoding="utf-8") as f:
                self.nodes = [json.loads(l) for l in f if l.strip()]
        if os.path.exists(self.edges_path):
            with open(self.edges_path, "r", encoding="utf-8") as f:
                self.edges = [json.loads(l) for l in f if l.strip()]
        self.id2pos = {n["id"]: i for i, n in enumerate(self.nodes)}
        self.vi.load(self.idx_path)
        self.vi.rebuild_bm25([n["text"] for n in self.nodes])

    def _save(self):
        # live append discipline: rewrite is fine at this scale; callers add one node
        # at a time, so each add persists immediately (§ live append).
        with open(self.nodes_path, "w", encoding="utf-8") as f:
            for n in self.nodes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        with open(self.edges_path, "w", encoding="utf-8") as f:
            for e in self.edges:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"seq": self._seq}, f)
        self.vi.save(self.idx_path)

    # ----------------------------------------------------------------- write path
    def _mk_id(self, kind, author, text):
        raw = f"{author}|{kind}|{self._seq}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def add_node(self, kind, author, text, payload=None, wire=True):
        """Append a node; auto-wire 'about' edges to recent nodes sharing a symbol."""
        if kind not in KINDS:
            raise ValueError(f"unknown node kind: {kind}")
        nid = self._mk_id(kind, author, text)
        self._seq += 1
        node = {"id": nid, "kind": kind, "author": author, "text": text,
                "payload": payload or {}}
        pos = len(self.nodes)
        self.nodes.append(node)
        self.id2pos[nid] = pos
        self.vi.add(text)
        if wire:
            self._self_wire(pos)
        self._save()
        return nid

    def _self_wire(self, pos, max_links=3):
        """Link node[pos] to up to max_links most-recent earlier nodes that share an
        identifier token. Zero LLM — pure overlap. Edge type 'about'."""
        toks = set(_ident_tokens(self.nodes[pos]["text"]))
        if not toks:
            return
        linked = 0
        for j in range(pos - 1, -1, -1):
            if linked >= max_links:
                break
            if toks & set(_ident_tokens(self.nodes[j]["text"])):
                self.link(self.nodes[pos]["id"], self.nodes[j]["id"], "about", save=False)
                linked += 1

    def link(self, src, dst, type="about", weight=1.0, save=True):
        if type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {type}")
        self.edges.append({"src": src, "dst": dst, "type": type, "weight": weight})
        if save:
            self._save()

    # ----------------------------------------------------------------- read path
    def neighbors(self, nid, type=None):
        out = []
        for e in self.edges:
            if e["src"] == nid and (type is None or e["type"] == type):
                out.append(e["dst"])
            elif e["dst"] == nid and (type is None or e["type"] == type):
                out.append(e["src"])
        return out

    def recent(self, kind=None, author=None, n=5):
        sel = [nd for nd in self.nodes
               if (kind is None or nd["kind"] == kind)
               and (author is None or nd["author"] == author)]
        return sel[-n:][::-1]   # newest first

    def get(self, nid):
        pos = self.id2pos.get(nid)
        return self.nodes[pos] if pos is not None else None

    def search(self, query, k=5, kinds=None):
        """Hybrid: vector + BM25 + graph-adjacency (neighbours of the top vector hit),
        fused via RRF. Returns up to k nodes best-first with per-signal detail."""
        if self.vi.ntotal == 0:
            return []
        vec_rank, vec_score = self.vi.vector_rank(query)
        vec_rank = [i for i in vec_rank if i < len(self.nodes)]
        bm_rank, bscores = self.vi.bm25_rank(query)

        # graph-adjacency ranking: positions of neighbours of the best vector hits
        adj_rank = []
        for seed in vec_rank[:3]:
            for nb in self.neighbors(self.nodes[seed]["id"]):
                p = self.id2pos.get(nb)
                if p is not None and p not in adj_rank:
                    adj_rank.append(p)

        fused = rrf([r for r in (vec_rank, bm_rank, adj_rank) if r])
        # gbrain SOURCE-TIER BOOST: multiply each fused score by its kind's authority, so
        # code/docs (fact/symbol) outrank session chatter (directive/result) and chatter
        # can't crowd out the real content (the duo2 retrieval bug).
        fused = {i: s * TIER.get(self.nodes[i]["kind"], 1.0) for i, s in fused.items()}
        order = sorted(fused, key=lambda i: fused[i], reverse=True)
        hits = []
        for i in order:
            nd = self.nodes[i]
            if kinds and nd["kind"] not in kinds:
                continue
            hits.append({**nd, "score": round(fused[i], 4),
                         "vec": round(vec_score.get(i, 0.0), 3),
                         "bm25": round(bscores[i] if i < len(bscores) else 0.0, 2),
                         "graph": i in adj_rank})
            if len(hits) >= k:
                break
        return hits

    def retrieve_content(self, query, k=3, max_chars=4000):
        """ON-DEMAND content retrieval (the scalable channel): return the CLEAN text of the
        top SOURCE-tier nodes (code/docs) for `query` — no `[kind/author]` labels, no chatter.
        This is what you feed a Reasoner for a BIG task instead of always-injecting the whole
        repo: pull only the relevant code/doc chunks when it needs them."""
        hits = self.search(query, k=k, kinds=SOURCE_KINDS)
        out, used = [], 0
        for h in hits:
            t = h["text"]
            if used + len(t) > max_chars:
                t = t[:max_chars - used]
            out.append(t)
            used += len(t)
            if used >= max_chars:
                break
        return "\n\n".join(out)

    def recall_history(self, query, k=3, max_chars=600):
        """History via RETRIEVAL done right: return only the top-k RELEVANT prior chatter
        (insights/results/corrections) for `query`, concise and deduped — NOT a raw dump of
        every recent node (that confused the 3B). One short line per item, no heavy labels."""
        hits = self.search(query, k=k * 2,
                           kinds=("directive", "correction", "result", "plan", "query"))
        seen, lines, used = set(), [], 0
        for h in hits:
            t = " ".join(h["text"].split())[:140]
            if t in seen:
                continue
            seen.add(t)
            tag = "tried" if h["kind"] in ("directive", "correction") else "saw"
            line = f"- {tag}: {t}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
            if len(lines) >= k:
                break
        return "\n".join(lines)

    def think(self, query, k=5):
        """gbrain `think`: synthesize a short answer from the top hits + a GAP note
        on what the graph does not yet contain. Not raw nodes — a digest."""
        hits = self.search(query, k=k)
        if not hits:
            return {"answer": "", "gap": "graph empty — no prior knowledge on this.",
                    "sources": []}
        lines = []
        for h in hits:
            lines.append(f"[{h['kind']}/{h['author']}] {h['text'][:160]}")
        kinds_present = {h["kind"] for h in hits}
        missing = [k for k in ("result", "fact") if k not in kinds_present]
        gap = ("graph has %s; missing %s" % (
            ", ".join(sorted(kinds_present)),
            ", ".join(missing) if missing else "nothing obvious"))
        return {"answer": "\n".join(lines), "gap": gap,
                "sources": [h["id"] for h in hits]}

    def window(self, query, recent_k=3, search_m=4, max_chars=2500):
        """§9.5 bounded, role-shaped context feed: newest `recent_k` tail nodes plus
        top `search_m` retrieved nodes, deduped, capped at max_chars. A brain never
        sees the whole graph — this is the only thing fed into a prompt."""
        seen, lines = set(), []
        for nd in self.recent(n=recent_k):
            if nd["id"] not in seen:
                seen.add(nd["id"])
                lines.append(f"[{nd['kind']}/{nd['author']}] {nd['text']}")
        for h in self.search(query, k=search_m):
            if h["id"] not in seen:
                seen.add(h["id"])
                lines.append(f"[{h['kind']}/{h['author']}] {h['text']}")
        return "\n".join(lines)[:max_chars]
