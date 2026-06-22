"""Smoke-test task decomposition: atomic stays 1 step; multi-bug splits and solves."""
import os
import shutil
import tempfile

from agent import tools
from agent.loop import run_planned
from agent.planner import plan

ATOMIC = "The file buggy.py has an off-by-one in count(). Fix it so the tests pass."

# A genuinely multi-step task: TWO independent bugs in TWO functions, one test suite.
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
        "functions. Inspect it, fix every bug by rewriting the file, and make "
        "run_tests pass.")


def main():
    print("=== 1) atomic task should yield ONE subtask ===")
    sub = plan(ATOMIC)
    print(f"  -> {len(sub)} subtask(s): {sub}")
    print(f"  {'PASS' if len(sub) == 1 else 'WARN: decomposed an atomic task'}\n")

    print("=== 2) two-bug task: decompose + solve in a shared sandbox ===")
    tmp = tempfile.mkdtemp(prefix="plan_")
    try:
        with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
            f.write(TWO_BUGS["buggy"])
        with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
            f.write(TWO_BUGS["test"])
        tools.set_sandbox(tmp)
        assert not tools.tests_pass(), "seed should fail"
        ok, steps = run_planned(TASK, max_steps=8, temperature=0.2, expose_kg=False)
        print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} in {steps} total steps")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))


if __name__ == "__main__":
    main()
