"""Hardcoded tools (no MCP yet). Every tool returns a short string result."""
import os
import subprocess
import sys

from .retriever import default_retriever

# Sandbox root: tools may only touch files under here. Configurable so the eval
# harness can point each problem at its own isolated sandbox + verifier.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))
TEST_FILE = "test_buggy.py"


def set_sandbox(root, test_file="test_buggy.py"):
    """Point the tools at a different sandbox dir + verifier (used by eval)."""
    global ROOT, TEST_FILE
    ROOT = os.path.abspath(root)
    TEST_FILE = test_file

# Lazily-built KG retriever (GitNexus today, swappable to our own SQLite KG).
_KG = None


def _kg():
    global _KG
    if _KG is None:
        _KG = default_retriever()
    return _KG


def _safe_path(path):
    full = os.path.abspath(os.path.join(ROOT, path) if not os.path.isabs(path) else path)
    if os.path.commonpath([full, ROOT]) != ROOT:
        raise ValueError(f"Path escapes sandbox: {path}")
    return full


def read_file(args):
    full = _safe_path(args["path"])
    if not os.path.exists(full):
        return f"ERROR: file not found: {args['path']}"
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def write_file(args, content):
    # Refuse degenerate writes — the model sometimes emits a placeholder like
    # "<code>" or an near-empty block, which would destroy the file (integrated-test
    # finding 2026-06-21). Fail safe: don't write, tell the model what went wrong.
    stripped = (content or "").strip()
    if len(stripped) < 20 or stripped.startswith("<") or "def " not in content:
        return ("ERROR: refused to write — content looks like a placeholder or is "
                "too short. Emit the FULL corrected file in the ```python block.")
    full = _safe_path(args["path"])
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"OK: wrote {len(content)} bytes to {args['path']}"


def run_tests(args=None):
    """Deterministic verifier: run the sandbox test, report pass/fail + output."""
    proc = subprocess.run(
        [sys.executable, TEST_FILE],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    out = (proc.stdout + proc.stderr).strip()
    status = "PASS" if proc.returncode == 0 else "FAIL"
    return f"[tests {status}, exit={proc.returncode}]\n{out}"


def kg_query(args):
    """Search the code knowledge graph for execution flows on a concept."""
    return _kg().query(args["query"])


def kg_context(args):
    """360-degree view of a symbol: callers, callees, processes."""
    return _kg().context(args["symbol"])


def kg_impact(args):
    """Blast radius: what breaks if you change a symbol."""
    return _kg().impact(args["symbol"], args.get("direction", "upstream"))


def tests_pass():
    proc = subprocess.run(
        [sys.executable, TEST_FILE],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    return proc.returncode == 0
