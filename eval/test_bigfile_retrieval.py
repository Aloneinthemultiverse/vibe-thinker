"""Genuinely exercise the CODE-RETRIEVAL scaling path on GPU.

A BIG file (25 functions, one bug) is chunked into per-function `fact` nodes on the graph.
The agent must RETRIEVE the relevant function via graph.retrieve_content() — NOT see the whole
file — reason about that chunk, fix it, and patch it back. Proves the scalable channel:
deliver only the relevant code chunk on demand instead of always-injecting the whole repo.

Run: python -m eval.test_bigfile_retrieval     (needs :8080 v12 + :8082 reasoner)
"""
import os
import re
import shutil
import tempfile
import time

from agent import llm, tools
from agent.graph import Graph
from agent.ingest import ingest_text
from agent.vecindex import default_embedder

# 24 correct, semantically-distinct filler functions + 1 buggy (running_total: stray +1)
FILLERS = [
    "def area_of_circle(r):\n    return 3.14159 * r * r",
    "def celsius_to_kelvin(c):\n    return c + 273.15",
    "def is_palindrome(s):\n    return s == s[::-1]",
    "def word_count(text):\n    return len(text.split())",
    "def factorial(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out",
    "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "def flatten(xss):\n    return [x for xs in xss for x in xs]",
    "def unique(xs):\n    return list(dict.fromkeys(xs))",
    "def second_largest(xs):\n    return sorted(set(xs))[-2]",
    "def vowel_count(s):\n    return sum(c in 'aeiou' for c in s.lower())",
    "def title_case(s):\n    return s.title()",
    "def mean(xs):\n    return sum(xs) / len(xs)",
    "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]",
    "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
    "def count_digits(n):\n    return len(str(abs(n)))",
    "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))",
    "def repeat(s, n):\n    return s * n",
    "def to_snake(s):\n    return s.replace(' ', '_').lower()",
    "def initials(name):\n    return ''.join(w[0] for w in name.split()).upper()",
    "def square_list(xs):\n    return [x * x for x in xs]",
    "def positive_only(xs):\n    return [x for x in xs if x > 0]",
    "def join_csv(xs):\n    return ','.join(str(x) for x in xs)",
    "def safe_div(a, b):\n    return a / b if b else 0",
]
BUGGY = "def running_total(xs):\n    return sum(xs) + 1   # BUG: stray +1"
TEST = ("from buggy import running_total\n"
        "assert running_total([1, 2, 3]) == 6\n"
        "assert running_total([]) == 0\n"
        "print('ok')\n")

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def build_file():
    return "\n\n".join(FILLERS[:12] + [BUGGY] + FILLERS[12:]) + "\n"


def split_functions(src):
    """Split a module into (name, code) per top-level def."""
    parts = re.split(r"\n(?=def )", src.strip())
    out = []
    for p in parts:
        m = re.match(r"def (\w+)", p)
        if m:
            out.append((m.group(1), p.strip()))
    return out


def patch_function(src, name, new_code):
    """Replace the top-level `def name(...)` block with new_code."""
    pat = re.compile(rf"(?m)^def {re.escape(name)}\(.*?(?=\n^def |\Z)", re.DOTALL)
    return pat.sub(new_code.strip() + "\n\n", src, count=1)


def main():
    if not (llm.healthy() and llm.reasoner_healthy()):
        print("ERROR: need :8080 + :8082 up"); return
    tmp = tempfile.mkdtemp(prefix="bigfile_")
    full = build_file()
    with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
        f.write(full)
    with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
        f.write(TEST)
    tools.set_sandbox(tmp)
    try:
        # seed each function as a fact node (the graph = code store)
        g = Graph(tempfile.mkdtemp(prefix="bigfile_g_"), embedder=default_embedder())
        for name, code in split_functions(full):
            ingest_text(g, code, source=name)

        failing = tools.run_tests()
        assert not tools.tests_pass(), "seed should fail"
        hint = next((l for l in failing.splitlines() if "Error" in l or "assert" in l), "")
        query = f"running total sum {hint}"

        # THE SCALING MOVE: retrieve only the relevant function, not the whole file
        retrieved = g.retrieve_content(query, k=2)
        print(f"whole file = {len(full)} chars | retrieved = {len(retrieved)} chars "
              f"({len(retrieved)/len(full):.0%} of file)")
        print(f"retrieved chunk:\n{retrieved[:300]}\n")
        assert len(retrieved) < 0.4 * len(full), "retrieval did not shrink context"
        assert "running_total" in retrieved, "retrieval missed the buggy function!"

        # Reasoner reasons over the RETRIEVED chunk only (+ the failing test)
        t0 = time.time()
        cot = llm.chat_reasoner(
            [{"role": "system", "content": "Expert Python debugger. Be exact about return values."},
             {"role": "user", "content":
              f"This function fails a test.\n\n```python\n{retrieved}```\n\n"
              f"Failing test:\n```python\n{TEST}```\n\nReason, then state the corrected function."}],
            temperature=0.4, max_tokens=2048)
        # Actor transcribes the corrected FUNCTION (not the whole file)
        reply = llm.chat(
            [{"role": "system", "content": "Output ONLY the corrected function in one ```python block."},
             {"role": "user", "content": f"Expert reasoning:\n{cot[-1800:]}\n\n"
              f"Original:\n```python\n{retrieved}```\n\nWrite the corrected function."}],
            temperature=0.2, max_tokens=512)
        blocks = [b.strip() for b in _CODE.findall(reply) if "def running_total" in b]
        if not blocks:
            print("actor produced no running_total block"); return
        fixed = blocks[0]
        patched = patch_function(full, "running_total", fixed)
        with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
            f.write(patched)
        ok = tools.tests_pass()
        print(f"patched running_total, ran tests in {time.time()-t0:.0f}s")
        print(f"\nVERDICT: {'PASS — retrieval-driven big-file fix works' if ok else 'FAIL'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))


if __name__ == "__main__":
    main()
