"""Run the seeded-bug benchmark and report effectiveness (pass@1, avg steps).

    python -m eval.run_eval [trials] [--kg]

Each problem runs `trials` times (default 3) in its OWN temp sandbox, with the bug
freshly seeded. KG tools are hidden by default (these are self-contained tasks);
pass --kg to expose them. Requires llama-server on 127.0.0.1:8080.
"""
import os
import shutil
import sys
import tempfile
import time

from agent import llm, tools
from agent.loop import run
if "--heldout" in sys.argv:
    from eval.heldout import PROBLEMS   # unseen-in-training generalization set
else:
    from eval.problems import PROBLEMS

TASK = ("The file buggy.py fails its test suite. Inspect it with read_file, fix the "
        "bug by rewriting the file with write_file, and make run_tests pass.")


def _seed(tmp, prob):
    with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
        f.write(prob["buggy"])
    with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
        f.write(prob["test"])


def main():
    trials = 3
    expose_kg = "--kg" in sys.argv
    # Episodic ('brain') memory is OFF in eval by DEFAULT: recalling a stored answer to
    # the very problem being graded is memorization-via-retrieval, which would fake the
    # generalization number. Opt in with --brain only for a deliberate brain-assist test.
    use_brain = "--brain" in sys.argv
    if not use_brain:
        os.environ["EPISODIC_OFF"] = "1"
    for a in sys.argv[1:]:
        if a.isdigit():
            trials = int(a)

    if not llm.healthy():
        print("ERROR: llama-server not reachable on 127.0.0.1:8080.")
        sys.exit(2)

    print(f"VibeThinker-OS effectiveness eval — {len(PROBLEMS)} problems x {trials} "
          f"trials, kg={'on' if expose_kg else 'off'}, "
          f"brain={'on' if use_brain else 'off'}, temp=0.2\n")
    rows = []
    t_start = time.time()
    for prob in PROBLEMS:
        passes, steps_on_pass = 0, []
        for t in range(trials):
            tmp = tempfile.mkdtemp(prefix=f"eval_{prob['name']}_")
            try:
                _seed(tmp, prob)
                tools.set_sandbox(tmp)
                if tools.tests_pass():
                    print(f"  [skip] {prob['name']} trial {t}: seed already passes?!")
                    continue
                try:
                    ok, steps = run(TASK, max_steps=8, expose_kg=expose_kg,
                                    temperature=0.2, use_episodic=use_brain)
                except Exception as e:   # one bad trial must not kill the suite
                    ok, steps = False, 0
                    print(f"  {prob['name']:16s} trial {t+1}/{trials}: ERROR {e}")
                passes += int(ok)
                if ok:
                    steps_on_pass.append(steps)
                print(f"  {prob['name']:16s} trial {t+1}/{trials}: "
                      f"{'PASS' if ok else 'FAIL'} ({steps} steps)")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        avg = sum(steps_on_pass) / len(steps_on_pass) if steps_on_pass else 0
        rows.append((prob["name"], passes, trials, avg))

    print("\n==== RESULTS ====")
    print(f"{'problem':18s} {'pass@1':>8s} {'avg steps':>10s}")
    tot_p = tot_t = 0
    for name, p, t, avg in rows:
        tot_p += p
        tot_t += t
        print(f"{name:18s} {p}/{t:<6d} {avg:>10.1f}")
    rate = tot_p / tot_t if tot_t else 0
    print(f"{'-'*38}")
    print(f"{'OVERALL':18s} {tot_p}/{tot_t} = {rate:.0%}")
    print(f"elapsed: {time.time()-t_start:.0f}s")
    # Restore default sandbox so other entry points are unaffected.
    tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))


if __name__ == "__main__":
    main()
