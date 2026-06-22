"""KG-driving demo: prove VibeThinker actually USES the knowledge-graph tools.

Unlike run_demo (which edits a sandboxed file), this task can only be answered by
querying the exact code knowledge graph of an indexed repo. No file editing, no
tests — success = the model drives kg_* tools and finishes with a grounded answer.

Run from project root, with GITNEXUS_REPO set:
    GITNEXUS_REPO=tinybench python -m agent.run_kg_demo
Requires llama-server on 127.0.0.1:8080 and `gitnexus` on PATH.
"""
import os
import sys

from . import llm
from .loop import run

REPO = os.environ.get("GITNEXUS_REPO", "tinybench")

TASK = (
    f"You are analyzing the '{REPO}' codebase, which you can ONLY inspect through "
    "the knowledge-graph tools (kg_query, kg_context, kg_impact) — you have no file "
    "access to it. Question: if someone changes the `Bench` class, what is the blast "
    "radius (what depends on it)? Use kg_impact to find out, optionally kg_context for "
    "detail, then finish with a one-sentence answer naming the impacted files and risk."
)


def main():
    if not llm.healthy():
        print("ERROR: llama-server not reachable on 127.0.0.1:8080. Start it first.")
        sys.exit(2)
    print(f"KG demo against repo: {REPO}")
    ok, steps = run(TASK, max_steps=6, require_tests=False)
    print(f"\n==== KG DEMO: {'SUCCESS' if ok else 'FAILURE'} in {steps} step(s) ====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
