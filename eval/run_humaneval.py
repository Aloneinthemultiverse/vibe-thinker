"""CoupleVibe on HumanEval -- a REAL, verifiable coding benchmark (164 problems, each with a
hidden test). This is the honest head-to-head axis vs Gemma: short self-contained code with a
checkable answer = CoupleVibe's sweet spot.

What makes it CoupleVibe and not just "run a 3B": the verify->fix loop. We generate, RUN the
test, and on failure feed the result back for one retry -- the test-time verification that lets
a small model beat a bigger single-shot model on tasks that have a checker.

  python -m eval.run_humaneval [N] [--noretry]

Reports pass@1 over the first N problems (default 20). Each candidate runs in a subprocess
with a timeout so a bad generation can't hang the run.
"""
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile

from agent import llm

DATA = os.path.join(os.path.dirname(__file__), "data", "HumanEval.jsonl.gz")


def _results_path(couple):
    name = "humaneval_results_couple.jsonl" if couple else "humaneval_results.jsonl"
    return os.path.join(os.path.dirname(__file__), "data", name)


def load(n):
    rows = [json.loads(l) for l in gzip.open(DATA, "rt", encoding="utf-8")]
    return rows if n is None else rows[:n]


def _load_done(path):
    """Read already-graded results so a crashed run can resume. Returns {task_id: record}."""
    done = {}
    if os.path.exists(path):
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["task_id"]] = r
    return done


def _append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<think>.*", "", text, flags=re.S)
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _reason(prompt, prior_fail=""):
    """Base VibeThinker-3B reasons toward the solution (raw, includes its CoT)."""
    usr = ("Complete this Python function. Return ONLY the full function definition (signature "
           "+ body), no explanation, no tests, no markdown.\n\n" + prompt)
    if prior_fail:
        usr += ("\n\nYour previous attempt FAILED its tests:\n" + prior_fail
                + "\nReturn a corrected full function.")
    # VibeThinker is a long-CoT model -- it thinks for thousands of tokens before the code.
    # A small cap truncates it before any code is emitted, so give it room.
    return llm.chat_reasoner([{"role": "system", "content": "You are an expert Python programmer."},
                              {"role": "user", "content": usr}], temperature=0.2, max_tokens=3584)


def _transcribe_v12(reasoner_raw):
    """THE COUPLE: v12 (Actor) transcribes the Reasoner's messy CoT into one clean function.
    This is duo.py's proven division. On pure codegen it may add variance -- that's the
    experiment. Falls back to the reasoner's own code if v12 returns nothing usable."""
    usr = ("A reasoning model produced this solution attempt (it may include thinking). "
           "Output ONLY the final, complete Python function definition it arrived at -- no "
           "explanation, no tests, no markdown.\n\n" + reasoner_raw[-4000:])
    out = llm.chat([{"role": "system", "content": "You extract the final clean code."},
                    {"role": "user", "content": usr}], temperature=0.1, max_tokens=1536)
    return extract_code(out)


def generate(prompt, prior_fail="", couple=False):
    raw = _reason(prompt, prior_fail)
    if not couple:
        return extract_code(raw)
    code = _transcribe_v12(raw)
    return code or extract_code(raw)   # fall back to reasoner code if v12 whiffs


def _prompt_header(prompt, entry):
    """Everything in the HumanEval prompt before `def entry` -- imports + any helpers the test
    relies on. Must be prepended to the candidate or imports like `from typing import List`
    are lost and the candidate NameErrors (a false fail)."""
    m = re.search(rf"^def\s+{re.escape(entry)}\s*\(", prompt, re.M)
    return prompt[:m.start()] if m else ""


def _ensure_signature(code, prompt, entry):
    """If the model returned only a body (no `def entry`), graft the prompt's signature on."""
    if re.search(rf"def\s+{re.escape(entry)}\s*\(", code):
        return code
    return prompt + "\n" + code


def run_test(code, test, entry, timeout=15, header=""):
    """True if code passes the HumanEval test. Runs isolated in a subprocess. `header` carries
    the prompt's imports/helpers so the candidate has the same context the test expects."""
    program = header + "\n" + code + "\n\n" + test + f"\n\ncheck({entry})\nprint('__PASSED__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return "__PASSED__" in p.stdout, (p.stderr or p.stdout)[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:300]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def solve_one(p, retry=True, couple=False):
    header = _prompt_header(p["prompt"], p["entry_point"])
    code = _ensure_signature(generate(p["prompt"], couple=couple), p["prompt"], p["entry_point"])
    ok, err = run_test(code, p["test"], p["entry_point"], header=header)
    used_retry = False
    if not ok and retry:
        used_retry = True
        code = _ensure_signature(generate(p["prompt"], prior_fail=err, couple=couple),
                                 p["prompt"], p["entry_point"])
        ok, err = run_test(code, p["test"], p["entry_point"], header=header)
    return ok, used_retry


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # `all` or omitted-with-many -> full 164. A number -> first N.
    if args and args[0].lower() in ("all", "full", "164"):
        n = None
    else:
        n = int(args[0]) if args else 20
    retry = "--noretry" not in sys.argv
    couple = "--couple" in sys.argv          # full Reasoner+v12 couple vs reasoner-only
    path = _results_path(couple)
    probs = load(n)
    done = _load_done(path)                   # resume: skip already-graded task_ids
    print(f"MODE: {'FULL COUPLE (Reasoner+v12)' if couple else 'Reasoner+verify'} -> {os.path.basename(path)}")
    for i, p in enumerate(probs, 1):
        tid = p["task_id"]
        if tid in done:
            r = done[tid]
            print(f"[{i}/{len(probs)}] {tid:<14} {'PASS' if r['ok'] else 'FAIL'} (cached)", flush=True)
            continue
        ok, used = solve_one(p, retry=retry, couple=couple)
        _append(path, {"task_id": tid, "ok": bool(ok), "used_retry": bool(used)})
        print(f"[{i}/{len(probs)}] {tid:<14} {'PASS' if ok else 'FAIL'}"
              f"{' (saved by retry)' if ok and used else ''}", flush=True)
    # Summarize over the problems in scope (from the checkpoint file).
    done = _load_done(path)
    scope = [p["task_id"] for p in probs]
    recs = [done[t] for t in scope if t in done]
    passed = sum(r["ok"] for r in recs)
    saves = sum(r["ok"] and r["used_retry"] for r in recs)
    single = passed - saves
    print(f"\n=== HumanEval pass@1: {passed}/{len(recs)} = {100*passed/max(1,len(recs)):.1f}% "
          f"({'with' if retry else 'NO'} verify-retry) ===")
    print(f"single-shot (no retry): {single}/{len(recs)} = {100*single/max(1,len(recs)):.1f}%")
    if retry:
        print(f"verify-loop rescued {saves} that single-shot would have failed")


if __name__ == "__main__":
    main()
