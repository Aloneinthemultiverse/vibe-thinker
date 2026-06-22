"""On-device ablation: how much does the HARNESS add over the raw model?

    python -m eval.ablation [trials] [--single] [--heldout|--multi]

--single  = NO harness: one model call shown the buggy file + test, extract the fix,
            verify once. Isolates the raw model on whatever GGUF is loaded on :8080.
(default) = FULL harness: run()/run_planned() with retry-hint + escalation + brain.

Swap the server's model between runs to compare base-3B vs v12 raw, then compare both
against the full-harness number. Same problems, same trials -> the deltas are real.
"""
import os
import re
import shutil
import sys
import tempfile
import time

from agent import llm, tools
from agent.loop import run, run_planned

SINGLE_SYS = ("You are a senior engineer. The file fails its tests. Reply with ONLY the "
              "complete corrected file inside one ```python code block — no prose.")
_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _problems():
    if "--multi" in sys.argv:
        from eval.heldout_multi import PROBLEMS
        return PROBLEMS, "multi"
    from eval.heldout import PROBLEMS
    return PROBLEMS, "heldout"


def single_shot(prob, tmp):
    msg = [{"role": "system", "content": SINGLE_SYS},
           {"role": "user", "content":
            f"buggy.py:\n```python\n{prob['buggy']}```\n\n"
            f"test_buggy.py:\n```python\n{prob['test']}```\n\nReturn the full corrected buggy.py."}]
    try:
        reply = llm.chat(msg, temperature=0.2)
    except Exception:
        return False
    m = _CODE.findall(reply)
    if not m:
        return False
    with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
        f.write(m[-1].strip() + "\n")
    return tools.tests_pass()


def main():
    trials = 1
    single = "--single" in sys.argv
    multi = "--multi" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            trials = int(a)
    if not single:
        os.environ["EPISODIC_OFF"] = "1"  # honest: no answer-key recall in ablation

    if not llm.healthy():
        print("ERROR: llama-server not reachable on :8080.")
        sys.exit(2)

    PROBLEMS, setname = _problems()
    mode = "RAW (single-shot, no harness)" if single else "FULL HARNESS"
    TASK = ("buggy.py fails its test suite. Inspect it, fix every bug by rewriting the "
            "file, and make run_tests pass.")
    print(f"ABLATION — {mode} | set={setname} | {len(PROBLEMS)}x{trials} trials\n")

    rows = []
    t0 = time.time()
    for prob in PROBLEMS:
        passes = 0
        for t in range(trials):
            tmp = tempfile.mkdtemp(prefix=f"abl_{prob['name']}_")
            try:
                with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
                    f.write(prob["buggy"])
                with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
                    f.write(prob["test"])
                tools.set_sandbox(tmp)
                if single:
                    ok = single_shot(prob, tmp)
                elif multi:
                    ok, _ = run_planned(TASK, max_steps=8, temperature=0.2, expose_kg=False)
                else:
                    ok, _ = run(TASK, max_steps=8, temperature=0.2, expose_kg=False,
                                use_episodic=False)
                passes += int(ok)
            except Exception as e:
                print(f"  {prob['name']:20s} t{t+1}: ERROR {e}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        rows.append((prob["name"], passes, trials))
        print(f"  {prob['name']:20s} {passes}/{trials}")

    tot_p = sum(r[1] for r in rows)
    tot_t = sum(r[2] for r in rows)
    print(f"\n==== {mode} | {setname} ====")
    print(f"OVERALL {tot_p}/{tot_t} = {tot_p/tot_t:.0%}   ({time.time()-t0:.0f}s)")
    tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))


if __name__ == "__main__":
    main()
