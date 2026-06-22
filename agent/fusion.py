"""Graph fusion: similarity finds the entry, structure expands it.

The plan's retrieval design, completed:
    natural-language query
      -> TurboQuant vector search over code chunks  (similarity, lossy)
      -> entry symbol + file
      -> GitNexus graph traversal on that symbol     (structure, exact)
      -> fused result (the hit AND what it's connected to)

Similarity gets you to the right neighborhood cheaply; the exact KG tells you the
truth about how it connects. Two indexes, one node identity (the symbol name/file).
"""
import os
import re

from .embed import embed, DIM
from .retriever import GitNexusRetriever
from .vectorstore import TurboQuantIndex, exact_topk, recall_at_k

_SRC_EXT = (".ts", ".tsx", ".js", ".py", ".go", ".java", ".rs")
# Top-level symbol declarations across the common languages we index.
_DECL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|async\s+)*"
    r"(?:class|function|def|interface|type|const|func)\s+([A-Za-z_]\w*)"
)


def chunk_repo(repo_path, max_lines=60, cap=400):
    """Walk source files, slice into symbol-anchored chunks."""
    chunks = []
    for root, _dirs, files in os.walk(repo_path):
        if any(p in root for p in ("node_modules", ".git", "dist", "build", ".gitnexus")):
            continue
        for fn in files:
            if not fn.endswith(_SRC_EXT):
                continue
            path = os.path.join(root, fn)
            try:
                lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
            except OSError:
                continue
            rel = os.path.relpath(path, repo_path).replace("\\", "/")
            marks = [(i, m.group(1)) for i, l in enumerate(lines)
                     for m in [_DECL.match(l)] if m]
            for j, (start, name) in enumerate(marks):
                end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
                end = min(end, start + max_lines)
                text = "\n".join(lines[start:end]).strip()
                if text:
                    chunks.append({"file": rel, "symbol": name, "text": text})
                if len(chunks) >= cap:
                    return chunks
    return chunks


class FusedRetriever:
    def __init__(self, repo_name, repo_path, bits=3):
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.bits = bits
        self.chunks = []
        self.index = None
        self.kg = GitNexusRetriever(repo=repo_name)

    def build(self):
        self.chunks = chunk_repo(self.repo_path)
        texts = [c["text"] for c in self.chunks]
        vecs = embed(texts)
        self.index = TurboQuantIndex(DIM, bits=self.bits).build(vecs)
        return self

    def search(self, query, k=5, expand=True):
        qv = embed(query)[0]
        hits = self.index.search(qv, k=k, rerank_n=64)
        entries = [self.chunks[i] for i in hits]
        result = {"query": query, "entries": entries}
        if expand and entries:
            # Structure expansion on the top entry's symbol (exact KG).
            top = entries[0]["symbol"]
            result["graph"] = self.kg.context(top)
        return result

    def validate_recall(self, k=10):
        """Recall@10 of the compressed index vs exact, on the REAL embeddings."""
        texts = [c["text"] for c in self.chunks]
        vecs = embed(texts)
        return recall_at_k(self.index, vecs, vecs, k=k, rerank_n=64)


def _demo():
    repo_name = os.environ.get("GITNEXUS_REPO", "tinybench")
    repo_path = os.environ.get("GITNEXUS_REPO_PATH",
                               r"C:\Users\Sujit Narrayan M\gh\tinybench")
    print(f"Fusion demo: repo={repo_name} path={repo_path}")
    fr = FusedRetriever(repo_name, repo_path, bits=3).build()
    print(f"indexed {len(fr.chunks)} symbol chunks; "
          f"compressed index = {fr.index.footprint_bytes()/1e3:.1f} KB")
    print(f"recall@10 (real embeddings, 3-bit) = {fr.validate_recall():.3f}")

    q = "run the benchmark tasks and collect timing results"
    res = fr.search(q, k=5)
    print(f"\nquery: {q}")
    print("vector entries (similarity):")
    for e in res["entries"]:
        print(f"  - {e['symbol']:20s} {e['file']}")
    g = res.get("graph", "")
    print(f"\ngraph expansion on top entry '{res['entries'][0]['symbol']}' (exact KG):")
    print("  " + (g[:300] if isinstance(g, str) else str(g)[:300]))


if __name__ == "__main__":
    _demo()
