"""Knowledge-graph retrieval for the orchestrator.

Design decision (2026-06-21): HYBRID. We adopt GitNexus (already indexing the
user's repos, exposes an exact code KG) as the structure substrate NOW, behind a
small abstract interface, so our OWN SQLite KG can slot in later for portability
without touching the loop. The interface is the contract; the backend is swappable.

GitNexus is reached via its CLI (subprocess), NOT via Claude Code's MCP binding —
our agent is a plain Python process talking to llama-server, so it cannot use the
editor's MCP tools. The CLI emits JSON on stdout for query/context/impact.
"""
import json
import os
import shutil
import subprocess


class Retriever:
    """Abstract structure-retrieval interface. Backends return compact text the
    3B can read directly. Three questions the orchestrator actually asks:
      - query(text)      : "what execution flows relate to this concept?"
      - context(symbol)  : "360 view of this symbol — callers, callees"
      - impact(symbol)   : "blast radius — what breaks if I change this?"
    """

    def query(self, text, limit=5):
        raise NotImplementedError

    def context(self, symbol):
        raise NotImplementedError

    def impact(self, symbol, direction="upstream"):
        raise NotImplementedError


def _compact_json(raw):
    """Return parsed JSON if the CLI emitted JSON, else the raw text trimmed."""
    raw = raw.strip()
    try:
        return json.dumps(json.loads(raw), separators=(",", ":"))
    except json.JSONDecodeError:
        return raw[:4000]


class GitNexusRetriever(Retriever):
    """Shell out to the `gitnexus` CLI. repo names come from `gitnexus list`."""

    def __init__(self, repo=None, exe="gitnexus", timeout=60):
        self.repo = repo or os.environ.get("GITNEXUS_REPO")
        self.exe = shutil.which(exe) or exe
        self.timeout = timeout

    def _run(self, *cmd):
        full = [self.exe, *cmd]
        if self.repo and "--repo" not in cmd:
            full += ["--repo", self.repo]
        try:
            proc = subprocess.run(
                full, capture_output=True, text=True, timeout=self.timeout,
            )
        except FileNotFoundError:
            return "ERROR: gitnexus CLI not found on PATH."
        except subprocess.TimeoutExpired:
            return "ERROR: gitnexus call timed out."
        out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        # Clean, short errors only — never leak a raw JS stack / file:// path that the
        # model might try to chase (integrated-test finding 2026-06-21).
        if proc.returncode != 0 or out.startswith("file://") or "backend.js" in out[:200]:
            hint = " (set repo: multiple repos are indexed)" if not self.repo else ""
            return (f"ERROR: gitnexus {cmd[0]} found nothing or failed{hint}. "
                    "This symbol/concept may not be in an indexed repo.")
        return _compact_json(out)

    def query(self, text, limit=5):
        return self._run("query", text, "--limit", str(limit))

    def context(self, symbol):
        return self._run("context", symbol)

    def impact(self, symbol, direction="upstream"):
        return self._run("impact", symbol, "--direction", direction)


class SqliteRetriever(Retriever):
    """Placeholder for our own portable KG (Step 4 'keep our own too' track).

    Intentionally unimplemented: it exists so the loop depends on the INTERFACE,
    not on GitNexus. When we build the SQLite code graph we implement these three
    methods and swap the backend in one line — no orchestrator changes.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path

    def _todo(self):
        raise NotImplementedError(
            "SqliteRetriever is the future portable backend; not built yet."
        )

    query = context = impact = lambda self, *a, **k: self._todo()


def default_retriever():
    """Pick the active backend. GitNexus today; SQLite later via env flag."""
    if os.environ.get("KG_BACKEND", "gitnexus").lower() == "sqlite":
        return SqliteRetriever(os.environ.get("KG_DB_PATH"))
    return GitNexusRetriever()
