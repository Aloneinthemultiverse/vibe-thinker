"""Generate SFT examples that teach TASK DECOMPOSITION (the planner skill).

Finding (2026-06-22): VibeThinker-3B will not decompose a multi-step task from a
zero/few-shot prompt — it echoes the task as a single step, exactly as it once refused
to emit tool calls. The cure is the same: show it the skill in SFT. Each example here
is a {system(planner), user(task), assistant(reason + ```plan [...])} triple in our
JSONL format, mergeable into sft_all via gen_synth/build_sft.

Crucially the set teaches BOTH directions:
  - atomic task  -> a ONE-element plan (must not over-decompose)
  - multi-step   -> the minimal ordered split
so the model learns WHEN to split, not just how.
"""
import json
import os

from agent.planner import _SYSTEM  # reuse the exact production planner system prompt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# (task, reasoning, [subtasks])
PLANS = [
    # ---- atomic: must stay single ----
    ("Fix the off-by-one in count() so the tests pass.",
     "One function, one bug. No decomposition needed.",
     ["Fix the off-by-one in count()"]),
    ("buggy.py's factorial() returns the wrong value for n=0. Fix it.",
     "A single missing base case in one function.",
     ["Add the n==0 base case to factorial()"]),
    ("Rename the variable `tmp` to `total` in sum_list().",
     "A single localized edit.",
     ["Rename tmp to total in sum_list()"]),
    ("The regex in validate_email() rejects valid addresses. Fix the pattern.",
     "One pattern to correct in one function.",
     ["Fix the regex pattern in validate_email()"]),
    ("Make is_palindrome() case-insensitive.",
     "A single behavioural tweak to one function.",
     ["Lowercase the input inside is_palindrome() before comparing"]),
    # ---- two independent bugs ----
    ("buggy.py has bugs in both to_celsius() and initials(). Fix every bug.",
     "Two unrelated functions each have a distinct bug; fix them separately.",
     ["Fix the formula bug in to_celsius()", "Fix initials() to use every word"]),
    ("parser.py fails: split_tokens() drops the last token and "
     "normalize() lowercases too aggressively. Fix both.",
     "Two named functions, two named defects.",
     ["Fix split_tokens() so it keeps the final token",
      "Fix normalize() to stop over-lowercasing"]),
    ("The math module has two bugs: gcd() loops forever on 0, and "
     "lcm() divides before multiplying. Repair both.",
     "Two independent algorithmic bugs.",
     ["Handle the zero case in gcd()", "Fix operator order in lcm()"]),
    # ---- feature + test (classic two-step) ----
    ("Add a retry decorator to api.py and add a unit test that proves it "
     "retries 3 times.",
     "Implement first, then verify — the test depends on the decorator existing.",
     ["Implement the retry decorator in api.py",
      "Write a test asserting it retries exactly 3 times"]),
    ("Add a /health endpoint and write a test for it.",
     "Build the endpoint, then test it.",
     ["Add the /health endpoint handler",
      "Write a test that asserts /health returns 200"]),
    ("Add input validation to register() and cover it with tests.",
     "Add the behaviour, then prove it.",
     ["Add input validation to register()",
      "Write tests for valid and invalid registration input"]),
    # ---- three-step pipelines (order matters) ----
    ("Read config.json, add a `timeout` field defaulting to 30, and update "
     "load_config() to use it.",
     "Inspect, then change the data, then wire it into the loader.",
     ["Inspect config.json and load_config()",
      "Add a timeout field defaulting to 30 in config.json",
      "Update load_config() to read and apply timeout"]),
    ("Refactor utils.py: extract the duplicated date-parsing logic into "
     "parse_date(), then make both callers use it, then run the tests.",
     "Create the shared helper, migrate callers, verify.",
     ["Extract the duplicated date logic into a new parse_date() helper",
      "Replace both inline copies with calls to parse_date()",
      "Run the test suite to confirm nothing broke"]),
    ("The CLI crashes on an empty file. Reproduce it with a test, fix the "
     "crash, then confirm the test passes.",
     "Reproduce, fix, confirm — TDD order.",
     ["Write a failing test that feeds an empty file to the CLI",
      "Fix the crash on empty input",
      "Run the test to confirm it now passes"]),
    # ---- atomic phrased to look big (anti over-decomposition) ----
    ("Completely fix the single typo in the error message string in log().",
     "Despite the emphatic wording, it is one trivial edit.",
     ["Fix the typo in the error message in log()"]),
    ("Do everything needed to make add(2,2) return 4 instead of 5.",
     "One arithmetic bug in one function.",
     ["Fix the arithmetic in add() so 2+2 returns 4"]),
]


def build_plans():
    rows = []
    for task, reason, subs in PLANS:
        plan_json = json.dumps(subs, ensure_ascii=False)
        assistant = f"{reason}\n```plan\n{plan_json}\n```"
        rows.append({"messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": task},
            {"role": "assistant", "content": assistant},
        ]})
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = build_plans()
    n_atomic = sum(1 for _, _, s in PLANS if len(s) == 1)
    out = os.path.join(DATA_DIR, "sft_plans.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} decomposition examples -> {out}")
    print(f"  atomic (1-step): {n_atomic} | multi-step: {len(rows) - n_atomic}")


if __name__ == "__main__":
    main()
