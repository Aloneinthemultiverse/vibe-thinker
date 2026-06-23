"""The REAL couple benchmark: agentic bug-fix (read -> reason -> edit -> verify -> retry).

Unlike HumanEval (pure codegen, where the Reasoner alone suffices), this exercises the whole
couple + harness: the agent must operate on a workspace, fix every bug, and turn a failing
test suite green end-to-end. This is where the two-brain design earns its keep.

  python -m eval.run_agentic          # HARD set (eval/agentic_hard.py)
  python -m eval.run_agentic --multi  # the easier heldout_multi set

Reports pass rate and how many were solved by the COUPLING (vs the single-brain fallback net).
"""
import os
import shutil
import sys
import tempfile
import time

os.environ["EPISODIC_OFF"] = "1"   # honesty: no answer-key recall

from agent import llm, tools
from agent.duo import solve

TASK = ("buggy.py fails its test suite — it has more than one bug across different functions. "
        "Inspect it, fix every bug by rewriting the file, and make run_tests pass.")


def _sandbox(buggy, test):
    tmp = tempfile.mkdtemp(prefix="agentic_")
    with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
        f.write(buggy)
    with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
        f.write(test)
    tools.set_sandbox(tmp)
    return tmp


def main():
    if not llm.healthy():
        print("ERROR: Actor server (:8080) not up."); sys.exit(2)
    if not llm.reasoner_healthy():
        print("ERROR: Reasoner server (:8082) not up."); sys.exit(2)
    if "--multi" in sys.argv:
        from eval.heldout_multi import PROBLEMS
        label = "heldout_multi"
    else:
        from eval.agentic_hard import PROBLEMS
        label = "agentic_hard"
    print(f"=== COUPLE agentic benchmark: {label} ({len(PROBLEMS)} tasks) ===")
    rows = []
    for p in PROBLEMS:
        tmp = _sandbox(p["buggy"], p["test"])
        t0 = time.time()
        try:
            ok, rounds, via = solve(TASK, verbose=False)
        except Exception as e:
            ok, rounds, via = False, 0, f"error:{str(e)[:40]}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rows.append((p["name"], ok, via))
        print(f"  {p['name']:22s} {'PASS' if ok else 'FAIL':4s}  via {via:9s} "
              f"({rounds} rnd, {time.time()-t0:.0f}s)", flush=True)
    n = sum(1 for _, ok, _ in rows if ok)
    c = sum(1 for _, ok, via in rows if ok and via == "coupling")
    print(f"\n=== {label}: {n}/{len(rows)} = {n/len(rows):.0%} passed "
          f"({c} via coupling, {n-c} via fallback) ===")


if __name__ == "__main__":
    main()
