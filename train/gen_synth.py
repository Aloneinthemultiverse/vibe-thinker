"""Generate a DIVERSE, test-verified SFT set and merge all sources into sft_all.jsonl.

Diversity is what drives generalization — ChatGPT gave 40 examples but only 5 distinct
functions. Here every problem is a different function/bug, and each fix is *executed*
against real asserts (buggy MUST fail, fix MUST pass) so nothing wrong enters the data.

Canonical trajectory taught (the discipline that fixes the overfit): each example is
read -> run_tests -> write fix -> finish (run_tests BEFORE writing, every time).

Sources merged:
  1. curated + harvested (from build_sft: PROBLEMS ideal trajectories + winning traces)
  2. these ~20 diverse synthetic problems (verified here)
  3. an external JSONL (e.g. ChatGPT output) passed as argv[1], structurally validated

Run:  python -m train.gen_synth [path/to/external.jsonl]
"""
import ast
import json
import os
import sys

from train.build_sft import (SYSTEM, TASK, _action, _write_turn, _obs,
                             build_curated, build_from_traces, DATA_DIR)

_FAIL = "[tests FAIL, exit=1]\nTraceback (most recent call last):\nAssertionError"
_PASS = "OK: wrote bytes to buggy.py\n[tests PASS, exit=0]\nok"

# (name, buggy, fixed, one-line diagnosis, asserts). Every function distinct.
SYNTH = [
    ("sum_list",
     "def sum_list(xs):\n    total = 1\n    for x in xs:\n        total += x\n    return total\n",
     "def sum_list(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n",
     "Accumulator must start at 0, not 1.",
     "assert sum_list([1,2,3])==6\nassert sum_list([])==0"),
    ("max_list",
     "def max_list(xs):\n    m = 0\n    for x in xs:\n        if x > m:\n            m = x\n    return m\n",
     "def max_list(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m\n",
     "Seeding max at 0 breaks all-negative inputs; seed with xs[0].",
     "assert max_list([-3,-1,-2])==-1\nassert max_list([4,9,2])==9"),
    ("count_vowels",
     "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
     "Must lowercase so uppercase vowels count too.",
     "assert count_vowels('AEIou')==5\nassert count_vowels('xyz')==0"),
    ("celsius_to_f",
     "def celsius_to_f(c):\n    return c * 9 / 5\n",
     "def celsius_to_f(c):\n    return c * 9 / 5 + 32\n",
     "Formula is missing the +32 offset.",
     "assert celsius_to_f(0)==32\nassert celsius_to_f(100)==212"),
    ("clamp",
     "def clamp(x, lo, hi):\n    return max(hi, min(lo, x))\n",
     "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
     "lo and hi are swapped in the max/min.",
     "assert clamp(5,0,10)==5\nassert clamp(-3,0,10)==0\nassert clamp(99,0,10)==10"),
    ("dedupe",
     "def dedupe(xs):\n    return list(set(xs))\n",
     "def dedupe(xs):\n    seen = []\n    for x in xs:\n        if x not in seen:\n            seen.append(x)\n    return seen\n",
     "set() loses order; preserve first-seen order.",
     "assert dedupe([3,1,3,2,1])==[3,1,2]"),
    ("second_largest",
     "def second_largest(xs):\n    return sorted(xs)[1]\n",
     "def second_largest(xs):\n    return sorted(set(xs))[-2]\n",
     "Need 2nd largest unique value, not 2nd smallest.",
     "assert second_largest([1,2,3,4])==3\nassert second_largest([9,8,7])==8"),
    ("is_anagram",
     "def is_anagram(a, b):\n    return sorted(a) == b\n",
     "def is_anagram(a, b):\n    return sorted(a) == sorted(b)\n",
     "Must compare sorted(a) to sorted(b), not raw b.",
     "assert is_anagram('listen','silent') is True\nassert is_anagram('a','b') is False"),
    ("safe_div",
     "def safe_div(a, b):\n    return a / b\n",
     "def safe_div(a, b):\n    if b == 0:\n        return 0\n    return a / b\n",
     "Guard division by zero.",
     "assert safe_div(6,2)==3\nassert safe_div(1,0)==0"),
    ("rotate_left",
     "def rotate_left(xs, k):\n    return xs[k:] + xs[k:]\n",
     "def rotate_left(xs, k):\n    return xs[k:] + xs[:k]\n",
     "Second slice should be xs[:k], not xs[k:].",
     "assert rotate_left([1,2,3,4],1)==[2,3,4,1]"),
    ("average",
     "def average(xs):\n    return sum(xs) // len(xs)\n",
     "def average(xs):\n    return sum(xs) / len(xs)\n",
     "Integer // truncates; use true division.",
     "assert average([1,2])==1.5"),
    ("first_word",
     "def first_word(s):\n    return s.split()[1]\n",
     "def first_word(s):\n    return s.split()[0]\n",
     "Index 1 is the second word; first is index 0.",
     "assert first_word('hello world')=='hello'"),
    ("count_evens",
     "def count_evens(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
     "def count_evens(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
     "%2==1 counts odds; evens are %2==0.",
     "assert count_evens([1,2,4,6])==3\nassert count_evens([1,3,5])==0"),
    ("gcd",
     "def gcd(a, b):\n    while b:\n        a, b = b, a // b\n    return a\n",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
     "Euclid uses a % b, not a // b.",
     "assert gcd(12,8)==4\nassert gcd(17,5)==1"),
    ("contains",
     "def contains(xs, t):\n    for x in xs:\n        return x == t\n    return False\n",
     "def contains(xs, t):\n    for x in xs:\n        if x == t:\n            return True\n    return False\n",
     "Early return after first element; only return True on a match.",
     "assert contains([1,2,3],3) is True\nassert contains([1,2],5) is False"),
    ("repeat",
     "def repeat(s, n):\n    return s + n\n",
     "def repeat(s, n):\n    return s * n\n",
     "Repeat is string * int, not string + int.",
     "assert repeat('ab',3)=='ababab'"),
    ("last_n",
     "def last_n(xs, n):\n    return xs[n:]\n",
     "def last_n(xs, n):\n    return xs[-n:]\n",
     "Last n elements is xs[-n:], not xs[n:].",
     "assert last_n([1,2,3,4,5],2)==[4,5]"),
    ("title_case",
     "def title_case(s):\n    return ' '.join(w.upper() for w in s.split())\n",
     "def title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
     "Title case capitalizes first letter only, not the whole word.",
     "assert title_case('hello world')=='Hello World'"),
    ("fib",
     "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-3)\n",
     "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n",
     "Recurrence is fib(n-1)+fib(n-2), not n-3.",
     "assert fib(7)==13\nassert fib(1)==1"),
    ("strip_spaces",
     "def strip_spaces(s):\n    return s.replace(' ', '_')\n",
     "def strip_spaces(s):\n    return s.replace(' ', '')\n",
     "Should remove spaces, not replace with underscore.",
     "assert strip_spaces('a b c')=='abc'"),
]


def _verify(name, buggy, fixed, asserts):
    """buggy MUST fail the asserts; fixed MUST pass. Returns True iff both hold."""
    def runs(code):
        ns = {}
        try:
            exec(code, ns)
            exec(asserts, ns)
            return True
        except Exception:
            return False
    ast.parse(buggy); ast.parse(fixed)        # syntax must be valid
    return (not runs(buggy)) and runs(fixed)


def synth_examples():
    out = []
    for name, buggy, fixed, diag, asserts in SYNTH:
        assert _verify(name, buggy, fixed, asserts), f"synth problem {name} failed verification"
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TASK: {TASK}"},
            {"role": "assistant",
             "content": f"Inspect buggy.py to locate the defect.\n{_action('read_file', {'path': 'buggy.py'})}"},
            _obs(buggy),
            {"role": "assistant",
             "content": f"Confirm the failure before editing.\n{_action('run_tests', {})}"},
            _obs(_FAIL),
            {"role": "assistant", "content": _write_turn(diag, "buggy.py", fixed)},
            _obs(_PASS),
            {"role": "assistant",
             "content": f"Verifier passes — done.\n{_action('finish', {'reason': 'tests pass'})}"},
        ]
        out.append({"messages": msgs})
    return out


def load_external(path):
    """Structurally validate an external JSONL (e.g. ChatGPT): valid JSON, starts system,
    ends assistant, has a write with valid-Python content. Drops anything malformed."""
    kept, dropped = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            m = r["messages"]
            assert m[0]["role"] == "system" and m[-1]["role"] == "assistant"
            for a in m:
                if a["role"] == "assistant" and "```python" in a["content"]:
                    code = a["content"].split("```python")[1].split("```")[0]
                    ast.parse(code)
            kept.append(r)
        except Exception:
            dropped += 1
    return kept, dropped


def _key(ex):
    return json.dumps(ex["messages"], sort_keys=True)


def main():
    curated = build_curated()
    harvested = build_from_traces()
    synth = synth_examples()
    external, dropped = ([], 0)
    if len(sys.argv) > 1:
        external, dropped = load_external(sys.argv[1])

    seen, merged = set(), []
    for src in (curated, synth, harvested, external):
        for ex in src:
            k = _key(ex)
            if k not in seen:
                seen.add(k); merged.append(ex)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "sft_all.jsonl"), "w", encoding="utf-8") as f:
        for ex in merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # diversity report
    import re
    fns = set()
    for ex in merged:
        for m in ex["messages"]:
            if m["role"] == "assistant" and "```python" in m["content"]:
                mt = re.search(r"def (\w+)", m["content"])
                if mt:
                    fns.add(mt.group(1))
    print(f"curated {len(curated)} | synth {len(synth)} (verified) | harvested {len(harvested)} "
          f"| external kept {len(external)} dropped {dropped}")
    print(f"-> data/sft_all.jsonl : {len(merged)} examples, {len(fns)} DISTINCT functions")


if __name__ == "__main__":
    main()
