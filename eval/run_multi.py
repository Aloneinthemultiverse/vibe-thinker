"""Harder MULTI-STEP benchmark: each problem has 2+ bugs; solved via run_planned.

    python -m eval.run_multi [trials] [--brain]

Default brain OFF -> honest baseline of plan + tool-calling + reasoning on compound
tasks. With --brain, an episodic memory pre-seeded with ANALOGOUS single-pattern fixes
(heldout_multi.SEEDS — never the compound answers) is made available, to measure whether
hybrid recall of related building blocks lifts the score FAIRLY. Same problems, same
trials, only the brain differs -> the delta is the brain's real contribution.
"""
import os
import shutil
import sys
import tempfile
import time

from agent import llm, tools
from agent import loop
from eval.heldout_multi import PROBLEMS, SEEDS

TASK = ("buggy.py fails its test suite — it has MORE THAN ONE bug across different "
        "functions. Inspect it, fix EVERY bug by rewriting the file, and make "
        "run_tests pass.")


def _seed_brain(store_dir):
    """Build an episodic brain pre-loaded with analogous building-block fixes."""
    from agent.episodic import EpisodicMemory
    mem = EpisodicMemory(store_dir=store_dir)
    if mem.index.ntotal == 0:                  # only seed an empty store
        for task, sol in SEEDS:
            mem.add(task, sol)
    return mem


def main():
    trials = 3
    use_brain = "--brain" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            trials = int(a)

    if not llm.healthy():
        print("ERROR: llama-server not reachable on 127.0.0.1:8080.")
        sys.exit(2)

    brain_dir = None
    if use_brain:
        os.environ.pop("EPISODIC_OFF", None)
        brain_dir = tempfile.mkdtemp(prefix="brain_multi_")
        loop._EPISODIC = _seed_brain(brain_dir)   # inject seeded brain (read-only-ish)
        print(f"[brain] seeded with {len(SEEDS)} analogous building-block examples")
    else:
        os.environ["EPISODIC_OFF"] = "1"

    print(f"VibeThinker-OS MULTI-STEP eval — {len(PROBLEMS)} compound problems x "
          f"{trials} trials, brain={'on' if use_brain else 'off'}, temp=0.2\n")

    rows = []
    t_start = time.time()
    for prob in PROBLEMS:
        passes = 0
        for t in range(trials):
            tmp = tempfile.mkdtemp(prefix=f"multi_{prob['name']}_")
            try:
                with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
                    f.write(prob["buggy"])
                with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
                    f.write(prob["test"])
                tools.set_sandbox(tmp)
                try:
                    ok, steps = loop.run_planned(TASK, max_steps=8, temperature=0.2,
                                                 expose_kg=False)
                except Exception as e:
                    ok, steps = False, 0
                    print(f"  {prob['name']:20s} trial {t+1}/{trials}: ERROR {e}")
                passes += int(ok)
                print(f"  {prob['name']:20s} trial {t+1}/{trials}: "
                      f"{'PASS' if ok else 'FAIL'} ({steps} steps)")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        rows.append((prob["name"], passes, trials))

    print("\n==== MULTI-STEP RESULTS ====")
    tot_p = tot_t = 0
    for name, p, t in rows:
        tot_p += p
        tot_t += t
        print(f"{name:22s} {p}/{t}")
    print("-" * 34)
    print(f"{'OVERALL':22s} {tot_p}/{tot_t} = {tot_p/tot_t:.0%}")
    print(f"elapsed: {time.time()-t_start:.0f}s")

    tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))
    if brain_dir:
        shutil.rmtree(brain_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
