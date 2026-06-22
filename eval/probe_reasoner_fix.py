"""Fast probe (ONE reasoner call): can the base Reasoner produce a PASSING fix for the
two-bug task when asked to emit the corrected FILE (not a prose directive)?

This validates the pivot (Reasoner-writes-the-fix) cheaply before wiring the full loop.
Run: python -m eval.probe_reasoner_fix
"""
import os
import re
import shutil
import tempfile
import time

from agent import llm, tools

BUGGY = ("def to_celsius(f):\n"
         "    return (f - 32) * 5 / 9 + 1   # bug: stray +1\n\n"
         "def initials(name):\n"
         "    return name[0]                # bug: only first word's initial\n")
TEST = ("from buggy import to_celsius, initials\n"
        "assert to_celsius(32) == 0\n"
        "assert to_celsius(212) == 100\n"
        "assert initials('ada lovelace') == 'AL'\n"
        "print('ok')\n")

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def main():
    tmp = tempfile.mkdtemp(prefix="probe_")
    with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
        f.write(BUGGY)
    with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
        f.write(TEST)
    tools.set_sandbox(tmp)
    try:
        msg = [{"role": "system", "content":
                "You are an expert Python engineer. Reason about the bugs, then OUTPUT the "
                "complete corrected file inside ONE ```python code block at the very end."},
               {"role": "user", "content":
                f"This file fails its tests. Fix EVERY bug.\n\nbuggy.py:\n```python\n{BUGGY}```\n\n"
                f"test_buggy.py:\n```python\n{TEST}```\n\nReason, then give the full corrected buggy.py."}]
        t0 = time.time()
        reply = llm.chat_reasoner(msg, temperature=0.4, max_tokens=4096)
        dt = time.time() - t0
        blocks = [b.strip() for b in _CODE.findall(reply) if b.strip()]
        print(f"reasoner reply: {len(reply)} chars, {len(blocks)} non-empty block(s), {dt:.0f}s")
        if not blocks:
            print("NO CODE BLOCK — pivot needs a different extraction.")
            print(reply[-500:])
            return
        # Pick the block that looks like the full corrected file: most 'def ', then longest.
        best = max(blocks, key=lambda b: (b.count("def "), len(b)))
        body = best + "\n"
        with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
            f.write(body)
        ok = tools.tests_pass()
        print(f"\nextracted fix:\n{body}")
        print(f"VERDICT: {'PASS — pivot works' if ok else 'FAIL — fix wrong'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))


if __name__ == "__main__":
    main()
