"""Phase D/E: run the dual-brain agent.

  python -m eval.run_duo          # Phase D: the canonical two-bug `initials` task
  python -m eval.run_duo --multi  # Phase E: heldout_multi, dual vs single

Escalation to a bigger model is OFF (no BIG_MODEL_ENDPOINT) — the point is to prove the
coupling itself solves what single-brain v12 fails, fully local.
"""
import os
import shutil
import sys
import tempfile
import time

os.environ["EPISODIC_OFF"] = "1"   # honesty: no answer-key recall (§9.6 posture)

from agent import llm, tools
from agent.duo import solve

TWO_BUGS = {
    "buggy": (
        "def to_celsius(f):\n"
        "    return (f - 32) * 5 / 9 + 1   # bug: stray +1\n\n"
        "def initials(name):\n"
        "    return name[0]                # bug: only first word's initial\n"
    ),
    "test": (
        "from buggy import to_celsius, initials\n"
        "assert to_celsius(32) == 0\n"
        "assert to_celsius(212) == 100\n"
        "assert initials('ada lovelace') == 'AL'\n"
        "print('ok')\n"
    ),
}
TASK = ("buggy.py fails its test suite — it has more than one bug across different "
        "functions. Inspect it, fix every bug by rewriting the file, and make run_tests pass.")


def _sandbox(buggy, test):
    tmp = tempfile.mkdtemp(prefix="duo_")
    with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
        f.write(buggy)
    with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
        f.write(test)
    tools.set_sandbox(tmp)
    return tmp


def phase_d():
    print("=== PHASE D: dual-brain on the two-bug `initials` task (no bigger model) ===")
    tmp = _sandbox(TWO_BUGS["buggy"], TWO_BUGS["test"])
    try:
        assert not tools.tests_pass(), "seed should fail"
        t0 = time.time()
        ok, rounds, via = solve(TASK)
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'} ({via}) in {rounds} rounds ({time.time()-t0:.0f}s)")
        met = ok and via == "coupling"
        print("SUCCESS CRITERION:",
              "MET — the COUPLING cracked it (no bigger model)" if met
              else f"NOT MET — solved via {via}" if ok else "NOT MET — failed")
        return met
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def phase_e():
    from eval.heldout_multi import PROBLEMS
    print(f"=== PHASE E: dual-brain on heldout_multi ({len(PROBLEMS)} problems) ===")
    rows = []
    for p in PROBLEMS:
        tmp = _sandbox(p["buggy"], p["test"])
        try:
            ok, rounds, via = solve(TASK, verbose=False)
        except Exception as e:
            print(f"  {p['name']:22s} ERROR {e}"); ok, rounds, via = False, 0, "error"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rows.append((p["name"], ok, via))
        print(f"  {p['name']:22s} {'PASS' if ok else 'FAIL':4s}  via {via}")
    n = sum(1 for _, ok, _ in rows if ok)
    c = sum(1 for _, ok, via in rows if ok and via == "coupling")
    print(f"\nDUAL-BRAIN heldout_multi: {n}/{len(rows)} = {n/len(rows):.0%} "
          f"({c} via coupling, {n-c} via fallback)")


def main():
    if not llm.healthy():
        print("ERROR: Actor server (:8080) not up."); sys.exit(2)
    if not llm.reasoner_healthy():
        print("ERROR: Reasoner server (:8082) not up — run smoke_reasoner first."); sys.exit(2)
    if "--multi" in sys.argv:
        phase_e()
    else:
        phase_d()
    tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))


if __name__ == "__main__":
    main()
