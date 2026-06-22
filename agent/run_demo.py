"""Entry point: reset the demo bug, then let the agent loop fix it.

Run from the project root:  python -m agent.run_demo
Requires llama-server running on 127.0.0.1:8080.
"""
import os
import sys

from . import llm, tools
from .loop import run

BUGGY = os.path.join(tools.ROOT, "buggy.py")

# The original seeded-bug version (so each run starts dirty).
SEED = '''def merge_intervals(intervals):
    # Merges overlapping intervals. intervals is a list of [start, end].
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] < last[1]:
            last[1] = current[1]
        else:
            merged.append(current)
    return merged
'''

TASK = ("The file buggy.py has a function merge_intervals with bug(s): the test "
        "suite fails. Use the tools to inspect it, fix the file, and make run_tests "
        "pass. The test command is run_tests.")


def main():
    if not llm.healthy():
        print("ERROR: llama-server not reachable on 127.0.0.1:8080. Start it first.")
        sys.exit(2)

    with open(BUGGY, "w", encoding="utf-8") as f:
        f.write(SEED)
    print("Seeded the bug. Verifier before agent:",
          "PASS" if tools.tests_pass() else "FAIL (expected)")

    ok, steps = run(TASK, max_steps=8)
    print(f"\n==== RESULT: {'SUCCESS' if ok else 'FAILURE'} in {steps} step(s) ====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
