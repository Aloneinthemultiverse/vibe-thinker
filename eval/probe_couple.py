"""Probe the WINNING coupling: Reasoner THINKS (messy but correct CoT) -> Actor (v12)
TRANSCRIBES that reasoning into a clean corrected file -> verify.

The earlier probe proved the base Reasoner reaches the right fix but rambles/truncates
before a clean code block. v12 is good at emitting clean code blocks. So couple them:
reasoner = the insight, actor = the clean hands. One reasoner call + one actor call.

Run: python -m eval.probe_couple
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
    tmp = tempfile.mkdtemp(prefix="couple_")
    with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
        f.write(BUGGY)
    with open(os.path.join(tmp, "test_buggy.py"), "w", encoding="utf-8") as f:
        f.write(TEST)
    tools.set_sandbox(tmp)
    try:
        # 1) REASONER thinks (long CoT, allowed to be messy)
        t0 = time.time()
        cot = llm.chat_reasoner(
            [{"role": "system", "content": "You are an expert Python debugger. Reason "
              "step by step about every bug and state the exact corrected code."},
             {"role": "user", "content":
              f"Fix every bug.\n\nbuggy.py:\n```python\n{BUGGY}```\n\n"
              f"test_buggy.py:\n```python\n{TEST}```"}],
            temperature=0.4, max_tokens=3072)
        t_reason = time.time() - t0
        print(f"[reasoner] {len(cot)} chars in {t_reason:.0f}s")

        # 2) ACTOR transcribes the reasoning into ONE clean corrected file
        t1 = time.time()
        actor = llm.chat(
            [{"role": "system", "content": "Output ONLY the complete corrected file inside "
              "one ```python code block. No prose."},
             {"role": "user", "content":
              f"An expert analyzed this file:\n\n{cot[-2500:]}\n\n"
              f"Original buggy.py:\n```python\n{BUGGY}```\n\n"
              "Write the FULL corrected buggy.py based on the expert's reasoning."}],
            temperature=0.2, max_tokens=1024)
        t_act = time.time() - t1
        blocks = [b.strip() for b in _CODE.findall(actor) if b.strip() and "def " in b]
        print(f"[actor] {len(actor)} chars in {t_act:.0f}s, {len(blocks)} code block(s)")
        if not blocks:
            print("actor produced no clean block:\n", actor[:400]); return
        body = max(blocks, key=lambda b: (b.count("def "), len(b))) + "\n"
        with open(os.path.join(tmp, "buggy.py"), "w", encoding="utf-8") as f:
            f.write(body)
        ok = tools.tests_pass()
        print(f"\nfinal file:\n{body}")
        print(f"VERDICT: {'PASS — coupling works!' if ok else 'FAIL'}  "
              f"(total {t_reason + t_act:.0f}s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tools.set_sandbox(os.path.join(os.path.dirname(__file__), "..", "demo"))


if __name__ == "__main__":
    main()
