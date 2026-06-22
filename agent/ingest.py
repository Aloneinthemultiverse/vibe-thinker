"""Document ingestion into the graph's source tier, via MarkItDown (MIT-licensed).

MarkItDown converts PDF / DOCX / PPTX / XLSX / HTML / CSV / etc. -> Markdown. We chunk that
markdown and add it as `fact` nodes so the Reasoner can RETRIEVE relevant doc context on
demand (the scalable channel) instead of always-injecting whole documents.

Lazy import: if markitdown isn't installed, ingest_text still works (no hard dependency).
    pip install 'markitdown[all]'
"""
import os


def _chunk(text, size=1200, overlap=150):
    """Split long markdown into overlapping chunks so retrieval is granular."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def to_markdown(path):
    """Convert a document to markdown via MarkItDown. Raises a clear error if not installed."""
    try:
        from markitdown import MarkItDown
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "markitdown not installed. Run: pip install 'markitdown[all]'") from e
    return MarkItDown().convert(path).text_content


def ingest_text(graph, text, source="doc"):
    """Add raw text/markdown to the graph as chunked source-tier `fact` nodes.
    Returns the node ids. No external deps — the testable core."""
    ids = []
    for j, ch in enumerate(_chunk(text)):
        ids.append(graph.add_node("fact", "system", f"DOC {source}#{j}:\n{ch}"))
    return ids


def ingest_document(graph, path):
    """Convert a document file -> markdown -> chunked `fact` nodes on the graph."""
    md = to_markdown(path)
    return ingest_text(graph, md, source=os.path.basename(path))
