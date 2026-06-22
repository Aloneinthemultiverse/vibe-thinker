"""Dual-brain orchestrator — coupled Reasoner + Actor, presented as one agent solve().

WHAT WORKS (validated 2026-06-23): the base Reasoner reaches the right fix in long CoT but
rambles / truncates before a clean code block; v12 (Actor) is excellent at emitting a single
clean code block. So couple them by ROLE:

  REASONER (base 3B, :8082, long CoT)  — THINKS: diagnoses the bug, states the fix.
  ACTOR    (v12, :8080, tool-tuned)    — HANDS: transcribes the reasoning into ONE clean
                                         corrected file via the write path; verifier checks.

They communicate through the shared Graph (blackboard): reasoner posts an `insight`, actor
posts a `result`. On failure, retry-with-hint (the proven move) feeds the exact failing
assertion back into the next reasoning round. Verifier (run_tests) anchors; no bigger model.

Guards: MAX_ROUNDS, no-progress detector (identical attempt -> stop), graceful degrade to
single-brain loop.run(). Everything local, $0.
"""
import os
import re
import tempfile

import numpy as np

from . import llm, tools
from .graph import Graph
from .vecindex import default_embedder

MAX_ROUNDS = int(os.environ.get("DUO_MAX_ROUNDS", "4"))
REASON_TOKENS = int(os.environ.get("DUO_REASON_TOKENS", "3072"))

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _hint(test_output):
    """Actionable nudge from verifier output (same idea as loop._hint_from_tests)."""
    if not test_output or "FAIL" not in test_output:
        return ""
    lines = [l.strip() for l in test_output.splitlines() if l.strip()]
    assertion = next((l for l in reversed(lines)
                      if "Error" in l or l.startswith("assert") or "!=" in l), "")
    return assertion[:200]


# ----------------------------------------------------------------- Reasoner (base 3B)
def _reason(task, current_src, failing):
    """Reasoner THINKS: produce CoT diagnosing the bug(s) and stating the exact fix.
    Allowed to ramble — the Actor distills it. Returns (cot_text, short_insight)."""
    hint = _hint(failing)
    usr = (f"{task}\n\nCurrent file:\n```python\n{current_src}```\n")
    if failing:
        usr += f"\nThe tests still FAIL. Latest output:\n{failing[:700]}\n"
    if hint:
        usr += (f"\nFocus: the failing check is `{hint}`. Reason about the EXACT expected "
                "value (types, case, order) and what the code must return to match it.")
    usr += "\nReason step by step about EVERY bug and state the exact corrected code."
    try:
        cot = llm.chat_reasoner(
            [{"role": "system", "content": "You are an expert Python debugger. Be precise "
              "about exact return values — types, case, ordering."},
             {"role": "user", "content": usr}],
            temperature=0.4, max_tokens=REASON_TOKENS)
    except Exception as e:
        return f"(reasoner error: {e})", "reasoner-unavailable"
    # short insight = the model's concluding lines (for the blackboard, not for parsing)
    tail = [l.strip() for l in cot.splitlines() if l.strip()][-3:]
    return cot, " ".join(tail)[:300]


# ----------------------------------------------------------------- Actor (v12)
def _best_block(reply):
    blocks = [b.strip() for b in _CODE.findall(reply) if b.strip() and "def " in b]
    if not blocks:
        return None
    return max(blocks, key=lambda b: (b.count("def "), len(b))) + "\n"


def _actor_write(task, cot, current_src, rel="buggy.py"):
    """Actor HANDS: transcribe the reasoning into ONE clean corrected file, write it,
    and verify. Returns (passed, body, test_output)."""
    try:
        reply = llm.chat(
            [{"role": "system", "content": "Output ONLY the complete corrected file inside "
              "one ```python code block. No prose."},
             {"role": "user", "content":
              f"Task: {task}\n\nAn expert analyzed the file:\n{cot[-2500:]}\n\n"
              f"Original file:\n```python\n{current_src}```\n\n"
              "Write the FULL corrected file based on the expert's reasoning."}],
            temperature=0.2, max_tokens=1024)
    except Exception as e:
        return False, None, f"[actor error: {e}]"
    body = _best_block(reply)
    if not body:
        return False, None, "[actor produced no code block]"
    res = tools.write_file({"path": rel}, body)
    if res.startswith("ERROR"):
        return False, body, res
    out = tools.run_tests()
    return tools.tests_pass(), body, out


# ----------------------------------------------------------------- the coupled loop
def _degrade(task):
    from .loop import run
    ok, steps = run(task, use_episodic=False, expose_kg=False)
    return ok, steps, "fallback"


def _target_file():
    """The single python file in the sandbox to fix (buggy.py by convention)."""
    return "buggy.py"


def solve(task, store_dir=None, verbose=True):
    """One-agent façade. Returns (ok, rounds, via): 'coupling' | 'fallback' | 'already'."""
    def say(*a):
        if verbose:
            msg = " ".join(str(x) for x in a)
            try:
                print(msg)
            except UnicodeEncodeError:   # Windows cp1252 console vs model's unicode
                print(msg.encode("ascii", "replace").decode())

    if not llm.reasoner_healthy():
        say("[duo] reasoner down -> single-brain run()")
        return _degrade(task)
    if tools.tests_pass():
        return True, 0, "already"

    store = store_dir or tempfile.mkdtemp(prefix="duo_shared_")
    g = Graph(store, embedder=default_embedder(), name="shared")
    emb = g.vi.embedder
    rel = _target_file()

    src = tools.read_file({"path": rel})
    if src.startswith("ERROR"):
        return _degrade(task)
    failing = tools.run_tests()
    last_body_vec = None

    for rnd in range(1, MAX_ROUNDS + 1):
        say(f"\n=== duo round {rnd}/{MAX_ROUNDS} ===")
        cot, insight = _reason(task, src, failing)
        g.add_node("directive", "reasoner", insight)
        say(f"  [reasoner] {len(cot)} chars | {insight[:90]}")

        passed, body, out = _actor_write(task, cot, src, rel)
        g.add_node("result", "actor",
                   (out.splitlines() or [""])[0][:140], payload={"passed": passed})
        say(f"  [actor] -> {'PASS' if passed else 'fail'}")

        if passed:
            say("  verifier PASSED -> success (coupling)")
            return True, rnd, "coupling"

        if body:
            # no-progress: identical corrected file two rounds running -> stop spinning
            bvec = emb.embed([body])[0]
            if last_body_vec is not None and float(np.dot(bvec, last_body_vec)) > 0.999:
                say("  [guard] no-progress (identical attempt) -> degrade")
                return _degrade(task)
            last_body_vec = bvec
            src = body          # next round reasons over the latest attempt
        failing = out

    if tools.tests_pass():
        return True, MAX_ROUNDS, "coupling"
    say("[duo] rounds exhausted -> single-brain net")
    return _degrade(task)
