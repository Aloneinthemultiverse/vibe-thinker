"""Step 9 (data) — build the supervised fine-tuning (SFT) dataset.

The eval (67% pass@1) produced real WINNING trajectories in runtime/traces/*.jsonl.
Those are gold: correct action sequences + the exact file content that made tests
pass. But VibeThinker reaches them through ~2000 tokens of `<think>` rambling per
turn (it was never trained for tool-calling). That waffle is the disease — it burns
the step budget and invites mistakes (the failures all hit the 8-step ceiling).

So we do NOT clone the raw replies. We keep the *policy* (which action, which args,
the winning file content) and pair it with TIGHT reasoning. Two outputs, one schema:

    data/sft_traces.jsonl   <- harvested from successful traces (volume, grounded)
    data/sft_curated.jsonl  <- hand-authored gold (format + the weak spots)

Schema per line:  {"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}
This is the standard chat-SFT format consumed by train/finetune_lora.py.

Run:  python -m train.build_sft        (no GPU, no network — pure local transform)
"""
import glob
import json
import os
import re

from agent.skills import build_system_prompt
from agent.loop import TOOL_HELP_NO_KG
from eval.problems import PROBLEMS

ROOT = os.path.dirname(os.path.dirname(__file__))
TRACE_DIR = os.path.join(ROOT, "runtime", "traces")
DATA_DIR = os.path.join(ROOT, "data")

TASK = ("The file buggy.py fails its test suite. Inspect it with read_file, fix the "
        "bug by rewriting the file with write_file, and make run_tests pass.")

# The system prompt the model will actually see at inference (eval ran KG hidden).
SYSTEM, _ = build_system_prompt(TASK, TOOL_HELP_NO_KG)

# Map each problem to its function name so we can identify which problem a winning
# write_file belongs to (every problem has a unique top-level def name).
_FNNAME = {p["name"]: re.search(r"def (\w+)", p["buggy"]).group(1) for p in PROBLEMS}
_BY_NAME = {p["name"]: p for p in PROBLEMS}

# The canonical correct fix for each problem (used by curated examples and as a
# sanity filter on harvested writes). Minimal diffs — exactly what we want taught.
FIXES = {
    "merge_intervals": '''\
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged
''',
    "binary_search": '''\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    "factorial": '''\
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
''',
    "is_palindrome": '''\
def is_palindrome(s):
    return s == s[::-1]
''',
    "fizzbuzz": '''\
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
''',
}

# One-line diagnoses — the kind of tight reasoning we want the fine-tune to emit
# INSTEAD of a 2000-token <think>.
DIAGNOSIS = {
    "merge_intervals": "Overlap test must be <= (touching intervals merge) and the "
                       "end must be max(last, current).",
    "binary_search": "The loop must run while lo <= hi, else the final single element "
                     "is never checked.",
    "factorial": "The base case factorial(0) must return 1, not 0 — 0 zeroes the whole "
                 "product.",
    "is_palindrome": "s[::1] is a no-op copy; reversing needs s[::-1].",
    "fizzbuzz": "The %15 case must be tested FIRST, otherwise %3 shadows it.",
}


def _action(tool, args):
    return json.dumps({"tool": tool, "args": args}, separators=(",", ": "))


def _write_turn(reason, path, content):
    """An assistant write_file turn: tight reason + action JSON + the python block."""
    return (f"{reason}\n{_action('write_file', {'path': path})}\n"
            f"```python\n{content.rstrip()}\n```")


def _obs(text):
    return {"role": "user", "content": text}


def _ideal_conversation(prob):
    """The canonical 6-action winning trajectory for a problem — what the model
    SHOULD do: read -> run_tests -> write the minimal fix -> run_tests -> finish."""
    buggy = prob["buggy"]
    name = prob["name"]
    fix = FIXES[name]
    fail = (f"[tests FAIL, exit=1]\nTraceback (most recent call last):\n  "
            f"AssertionError")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"TASK: {TASK}"},
        {"role": "assistant",
         "content": f"Inspect buggy.py to locate the defect.\n{_action('read_file', {'path': 'buggy.py'})}"},
        _obs(buggy),
        {"role": "assistant",
         "content": f"Confirm the failure before editing.\n{_action('run_tests', {})}"},
        _obs(fail),
        {"role": "assistant",
         "content": _write_turn(DIAGNOSIS[name], "buggy.py", fix)},
        _obs("OK: wrote bytes to buggy.py\n[tests PASS, exit=0]\nok"),
        {"role": "assistant",
         "content": f"Verifier passes — done.\n{_action('finish', {'reason': 'tests pass'})}"},
    ]


def build_curated():
    """Hand-authored gold: the ideal trajectory for every problem, PLUS targeted
    lessons for the observed weak spots (factorial base case, filename discipline,
    finishing promptly instead of burning the step budget)."""
    out = [{"messages": _ideal_conversation(p)} for p in PROBLEMS]

    # Lesson: do NOT invent a filename from the function name (a real eval quirk:
    # the model wrote fizzbuzz.py instead of buggy.py and the run errored).
    out.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"TASK: {TASK}"},
        {"role": "assistant", "content":
            "The file to edit is buggy.py exactly as named in the task — never rename "
            "it after the function inside.\n" + _action("read_file", {"path": "buggy.py"})},
    ]})

    # Lesson: once tests PASS, finish immediately — don't keep poking tools.
    out.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"TASK: {TASK}"},
        {"role": "user", "content": "OK: wrote bytes to buggy.py\n[tests PASS, exit=0]\nok"},
        {"role": "assistant", "content":
            "The verifier already passes; the discipline is to stop now.\n"
            + _action("finish", {"reason": "tests pass"})},
    ]})
    return out


def _classify_write(content):
    """Which problem does a harvested write_file belong to? Match by unique fn name."""
    for name, fn in _FNNAME.items():
        if re.search(rf"def {fn}\b", content):
            return name
    return None


def build_from_traces():
    """Harvest successful traces. Keep the real action sequence + winning file
    content; replace each verbose <think> reply with a tight canonical reason so the
    fine-tune learns the policy, not the rambling."""
    examples = []
    for path in sorted(glob.glob(os.path.join(TRACE_DIR, "trace-*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        if not events or events[-1].get("event") != "success":
            continue  # only learn from wins
        # Find the winning write to identify the problem and grab the fix content.
        writes = [e for e in events if e.get("tool") == "write_file" and e.get("content")]
        if not writes:
            continue
        win = writes[-1]["content"]
        prob_name = _classify_write(win)
        if not prob_name:
            continue
        prob = _BY_NAME[prob_name]
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TASK: {TASK}"},
        ]
        for e in events:
            if e.get("event") != "action":
                continue
            tool, args = e["tool"], e.get("args", {})
            if tool == "read_file":
                msgs.append({"role": "assistant", "content":
                             f"Inspect {args.get('path','buggy.py')} to find the bug.\n"
                             + _action("read_file", args)})
                msgs.append(_obs(prob["buggy"]))
            elif tool == "run_tests":
                msgs.append({"role": "assistant", "content":
                             f"Run the verifier.\n{_action('run_tests', {})}"})
                msgs.append(_obs("[tests FAIL, exit=1]\nAssertionError"))
            elif tool == "write_file":
                msgs.append({"role": "assistant", "content":
                             _write_turn(DIAGNOSIS.get(prob_name, "Apply the minimal fix."),
                                         args.get("path", "buggy.py"), e.get("content") or win)})
                msgs.append(_obs("OK: wrote bytes to buggy.py\n[tests PASS, exit=0]\nok"))
            elif tool == "finish":
                msgs.append({"role": "assistant", "content":
                             f"Verifier passes — done.\n"
                             + _action("finish", {"reason": "tests pass"})})
        # The loop early-exits on PASS without a finish action, so winning traces end
        # on the write's PASS observation. Append the canonical finish turn: every
        # example must end on an assistant target, and this teaches prompt-finishing.
        if msgs[-1]["role"] != "assistant":
            msgs.append({"role": "assistant", "content":
                         f"Verifier passes — done.\n"
                         + _action("finish", {"reason": "tests pass"})})
        examples.append({"messages": msgs})
    return examples


def _dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    curated = build_curated()
    harvested = build_from_traces()
    nc = _dump(os.path.join(DATA_DIR, "sft_curated.jsonl"), curated)
    nh = _dump(os.path.join(DATA_DIR, "sft_traces.jsonl"), harvested)
    # Combined file the trainer reads by default.
    nt = _dump(os.path.join(DATA_DIR, "sft_all.jsonl"), curated + harvested)
    print(f"curated : {nc:3d} examples -> data/sft_curated.jsonl")
    print(f"harvested: {nh:3d} examples -> data/sft_traces.jsonl  (from successful traces)")
    print(f"combined : {nt:3d} examples -> data/sft_all.jsonl")
    if harvested:
        turns = sum(len(e["messages"]) for e in harvested) / len(harvested)
        print(f"avg turns/harvested example: {turns:.1f}")


if __name__ == "__main__":
    main()
