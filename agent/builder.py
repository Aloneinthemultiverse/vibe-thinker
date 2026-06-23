"""CoupleVibe project builder: scale a 2x3B past single-file by DECOMPOSING a big task
into many small, calc.py-sized steps and doing them one at a time.

Why this works where duo2.py failed: duo2 let the BASE reasoner decompose freely and it
hallucinated subgoals (it won't follow a format). Here the division plays to each brain's
real strength, with a grammar guaranteeing a well-formed plan:

  - v12 (Actor, fine-tuned for STRUCTURE) decomposes -> a JSON step list. A GBNF grammar
    forces valid JSON, so the plan can't be malformed (duo2's exact failure).
  - base Reasoner (strong CODER per LiveCodeBench) generates each file's full content.

Each step is one whole file -- the single-file shape we PROVED works (calc.py). The harness
orchestrates; the model only ever does small pieces.
"""
import json
import os
import re
import subprocess

from . import llm

# GBNF: the plan must be a JSON array of {"file": str, "description": str}. Grammar makes a
# malformed plan impossible -- the thing that made duo2 wander.
_PLAN_GRAMMAR = (
    'root   ::= "[" ws item ( ws "," ws item )* ws "]"\n'
    'item   ::= "{" ws "\\"file\\"" ws ":" ws string ws "," ws '
    '"\\"description\\"" ws ":" ws string ws "}"\n'
    'string ::= "\\"" ( [^"\\\\] | "\\\\" . )* "\\""\n'
    'ws     ::= [ \\t\\n]*\n'
)

# A file is a STUB (not real content) if it's tiny or full of placeholder markers. We reject
# these so the builder doesn't "succeed" with `... CSS ...` skeletons.
_STUB_MARKERS = ("... css ...", "... js ...", "// javascript code", "// your code",
                 "# your code", "todo: implement", "...rest", "// ...", "/* ... */",
                 "rest of", "implementation here", "code here")


def _strip_fence(s):
    # Drop the base Reasoner's chain-of-thought first -- it leaks <think>...</think> into
    # file content otherwise (a real bug we hit on index.html).
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = re.sub(r"<think>.*", "", s, flags=re.S)  # unclosed think (truncated CoT)
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", s, re.S)
    return (m.group(1) if m else s).strip()


def _syntax_error(path, content):
    """Return a syntax-error string for a JS/Python file, or '' if it's fine / uncheckable.
    Uses node --check (JS) and py_compile (Python) -- the verifiers we already rely on."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".js", ".mjs"):
            p = subprocess.run(["node", "--check", path], capture_output=True, text=True, timeout=20)
            return "" if p.returncode == 0 else (p.stderr or p.stdout)[:600]
        if ext == ".py":
            p = subprocess.run(["python", "-m", "py_compile", path], capture_output=True,
                               text=True, timeout=20)
            return "" if p.returncode == 0 else (p.stderr or p.stdout)[:600]
    except Exception:
        return ""  # no checker available -> don't block
    return ""


def decompose(task, max_steps=8):
    """v12 (structure) -> ordered list of {file, description} steps. Grammar-guaranteed JSON.
    Falls back to a single-file plan if parsing somehow fails."""
    usr = (f"Break this build task into an ordered list of small steps, ONE FILE PER STEP.\n"
           f"TASK: {task}\n\n"
           "Output a JSON array. Each element: "
           '{"file": "<relative path>", "description": "<what this file must contain>"}.\n'
           "Keep it minimal and concrete. No prose outside the JSON.")
    try:
        raw = llm.chat([{"role": "system", "content": "You plan software builds as JSON."},
                        {"role": "user", "content": usr}],
                       temperature=0.2, max_tokens=1024, grammar=_PLAN_GRAMMAR)
        steps = json.loads(_strip_fence(raw))
    except Exception:
        steps = []
    clean = [s for s in steps
             if isinstance(s, dict) and s.get("file") and s.get("description")][:max_steps]
    if not clean:
        # Last resort: a single index file (still goes through the proven content path).
        clean = [{"file": "index.html", "description": task}]
    return clean


def generate_file(task, step, prior_files):
    """Reasoner (strong coder) writes the COMPLETE content of one file. prior_files lists the
    other files in the project so it can reference them correctly (e.g. <script src=...>)."""
    others = ", ".join(prior_files) or "(none yet)"
    usr = (f"Project task: {task}\n\n"
           f"Write the COMPLETE, WORKING contents of the file `{step['file']}`.\n"
           f"This file must: {step['description']}\n"
           f"Other files in this project: {others}\n\n"
           "Output ONLY the raw file content. No explanation, no markdown fences, no "
           "placeholder comments -- write every line of real, working code.")
    r = llm.chat_reasoner([{"role": "system", "content": "You are an expert programmer."},
                           {"role": "user", "content": usr}], temperature=0.2, max_tokens=4096)
    return _strip_fence(r)


def _is_stub(content):
    if len(content.strip()) < 60:
        return True
    low = content.lower()
    return any(m in low for m in _STUB_MARKERS)


def build(task, outdir, verbose=True):
    """Decompose -> generate each file -> write. Returns a report dict.
    Regenerates once if a file comes back as a stub (the web-app failure mode)."""
    os.makedirs(outdir, exist_ok=True)
    steps = decompose(task)
    if verbose:
        print(f"PLAN ({len(steps)} steps):")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s['file']} -- {s['description'][:70]}")
    written, stubs = [], []
    for s in steps:
        content = generate_file(task, s, written)
        if _is_stub(content):  # one retry with a sterner nudge
            content = generate_file(task, {**s, "description": s["description"]
                                           + " (write FULL implementation, no placeholders)"},
                                    written)
        path = os.path.join(outdir, s["file"])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        # VERIFY -> FIX (duo.py's proven pattern): if the file has a syntax error, feed the
        # exact error back to the Reasoner once and rewrite. Lifts raw 3B code over the line.
        err = _syntax_error(path, content)
        if err:
            if verbose:
                print(f"  {s['file']} syntax error -> asking Reasoner to fix")
            fixed = generate_file(task, {**s, "description": s["description"]
                                         + f"\nThe previous attempt had this error, FIX IT:\n{err}"},
                                  written)
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed)
            content, err = fixed, _syntax_error(path, fixed)
        n = len(content.splitlines())
        bad = _is_stub(content) or bool(err)
        (stubs if bad else written).append(s["file"])
        if verbose:
            flag = " [STUB]" if _is_stub(content) else (" [SYNTAX ERR]" if err else "")
            print(f"  wrote {s['file']} ({n} lines){flag}")
    return {"steps": [s["file"] for s in steps], "written": written, "stubs": stubs,
            "ok": not stubs and bool(written)}
