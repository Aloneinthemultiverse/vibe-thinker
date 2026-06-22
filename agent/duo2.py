"""Dual-brain v2 — the graph-DRIVEN, multi-step, tool-using coupling (Manus-like).

Difference from duo.py (which only transcribes a fix):
  1. THE GRAPH IS THE CONTEXT BUS. Both brains are prompted from g.window(...) — context
     flows through the shared graph, not side variables. Every action/result is posted LIVE.
  2. THE ACTOR USES REAL TOOLS, MULTI-STEP. Not one transcribe: v12 emits one action per
     step (read_file -> write_file -> run_tests ...) via the action protocol, several steps
     per subgoal.
  3. GLOBAL + LOCAL THINK. Tier-G: Reasoner decomposes the task into subgoals ONCE. Tier-L:
     Reasoner posts a directive (insight, from free-form CoT) per subgoal and a correction on
     failure. Both read the graph.

Verifier (run_tests) anchors; graceful degrade to single-brain run(). duo.py stays the proven
baseline; this is the research path. All local, $0.
"""
import os
import re
import tempfile

import numpy as np

from . import llm, tools
from .graph import Graph
from .protocol import parse_action, ParseError
from .skills import build_system_prompt
from .vecindex import default_embedder

MAX_SUBGOALS = int(os.environ.get("DUO2_MAX_SUBGOALS", "4"))
STEP_BUDGET = int(os.environ.get("DUO2_STEP_BUDGET", "4"))      # actor tool-steps per subgoal
MAX_CORRECTIONS = int(os.environ.get("DUO2_MAX_CORRECTIONS", "2"))
REASON_TOKENS = int(os.environ.get("DUO2_REASON_TOKENS", "2048"))

ACTOR_HELP = (
    "Available tools:\n"
    '- read_file   args: {"path": "<file>"}\n'
    '- write_file  args: {"path": "<file>"}  + FULL file in a ```python block\n'
    "- run_tests   args: {}\n"
    '- finish      args: {"reason": "<why>"}  (only after tests PASS)\n'
    "Emit EXACTLY ONE action as JSON; for write_file include the full file in a ```python block.")
TOOLSET = {"read_file", "write_file", "run_tests", "finish"}

_FIX_RE = re.compile(r"FIX:\s*(.+)", re.S | re.I)
_ECHO = re.compile(r"<[^>]*sentence|<[^>]*fix|<[^>]*bug|comma-separated|INSIGHT:", re.I)


def _hint(test_output):
    if not test_output or "FAIL" not in test_output:
        return ""
    lines = [l.strip() for l in test_output.splitlines() if l.strip()]
    return next((l for l in reversed(lines)
                 if "Error" in l or l.startswith("assert") or "!=" in l), "")[:200]


def _extract_insight(reply, fallback):
    m = _FIX_RE.search(reply)
    if m:
        cand = m.group(1).strip()
    else:
        lines = [l.strip() for l in reply.splitlines()
                 if l.strip() and not _ECHO.search(l) and not l.strip().startswith("#")]
        cand = " ".join(lines[-3:]) if lines else ""
    return (fallback if (not cand or _ECHO.search(cand)) else cand)[:400]


# --------------------------------------------------------------- Reasoner (base, :8082)
def _reason_plan(task, ctx):
    """Tier-G global think: decompose into ordered subgoals (once)."""
    usr = (f"{task}\n\nWhat is known so far:\n{ctx}\n\n"
           "Reason briefly, then list the concrete subgoals to fix this, one per line "
           "as 'SUBGOAL: <goal>'. If it is a single fix, give one subgoal.")
    try:
        reply = llm.chat_reasoner(
            [{"role": "system", "content": "You are the Reasoner. Decompose the work."},
             {"role": "user", "content": usr}], temperature=0.5, max_tokens=REASON_TOKENS)
    except Exception:
        return [task]
    subs = [m.strip() for m in re.findall(r"SUBGOAL:\s*(.+)", reply, re.I)]
    return subs[:MAX_SUBGOALS] or [task]


def _reason_directive(task, subgoal, ctx, failing=None):
    """Tier-L local think: insight for the current subgoal (free-form CoT -> FIX line)."""
    usr = (f"TASK: {task}\nSUBGOAL: {subgoal}\n\nBLACKBOARD (recent + relevant):\n{ctx}\n")
    if failing:
        usr += f"\nLatest test failure: {_hint(failing)}\n"
    usr += ("\nReason step by step about the exact bug and fix (exact values: types, case, "
            "order). End with one line 'FIX: <the concrete change in plain words>'.")
    try:
        reply = llm.chat_reasoner(
            [{"role": "system", "content": "You are an expert Python debugger guiding an "
              "Actor. Be precise about exact return values."},
             {"role": "user", "content": usr}], temperature=0.4, max_tokens=REASON_TOKENS)
    except Exception as e:
        return f"(reasoner error {e})"
    return _extract_insight(reply, fallback=subgoal)


# --------------------------------------------------------------- Actor (v12, :8080)
def _actor_action(task, subgoal, insight, ctx, failing):
    """One Actor tool-step via the action protocol. Returns (tool, args, content) or None."""
    framed = (f"{task}\n\nCURRENT SUBGOAL: {subgoal}\nREASONER INSIGHT: {insight}\n\n"
              f"BLACKBOARD:\n{ctx}\n")
    if failing:
        framed += f"\nLATEST TEST OUTPUT:\n{failing[:600]}\n"
    sys_prompt, _ = build_system_prompt(framed, ACTOR_HELP)
    try:
        reply = llm.chat([{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": framed}], temperature=0.2, max_tokens=1024)
    except Exception:
        return None
    try:
        tool, args, content = parse_action(reply)
    except ParseError:
        return None
    if tool not in TOOLSET:
        return None
    return tool, args, content


def _run_tool(tool, args, content):
    try:
        if tool == "read_file":
            return tools.read_file(args)
        if tool == "write_file":
            return tools.write_file(args, content) + "\n" + tools.run_tests()
        if tool == "run_tests":
            return tools.run_tests()
        if tool == "finish":
            return "[finish] " + ("PASS" if tools.tests_pass() else "FAIL")
    except Exception as e:
        return f"ERROR: tool {tool} failed: {e}"
    return f"ERROR: unhandled {tool}"


# --------------------------------------------------------------- the coupled loop
def _degrade(task):
    from .loop import run
    ok, steps = run(task, use_episodic=False, expose_kg=False)
    return ok, steps, "fallback"


def solve(task, store_dir=None, verbose=True):
    """Graph-driven multi-step tool agent. Returns (ok, steps, via)."""
    def say(*a):
        if verbose:
            msg = " ".join(str(x) for x in a)
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode())

    if not llm.reasoner_healthy():
        say("[duo2] reasoner down -> single-brain"); return _degrade(task)
    if tools.tests_pass():
        return True, 0, "already"

    store = store_dir or tempfile.mkdtemp(prefix="duo2_")
    g = Graph(store, embedder=default_embedder(), name="shared")

    # Tier-G global think (once), context from the graph
    failing = tools.run_tests()
    g.add_node("result", "system", "initial: " + (failing.splitlines() or [""])[0][:120],
               payload={"passed": False})
    subgoals = _reason_plan(task, g.window(task))
    g.add_node("plan", "reasoner", "PLAN: " + " | ".join(subgoals),
               payload={"subgoals": subgoals})
    say(f"[duo2] plan: {len(subgoals)} subgoal(s)")
    for i, s in enumerate(subgoals, 1):
        say(f"   {i}. {s[:80]}")

    total_steps = 0
    for si, subgoal in enumerate(subgoals, 1):
        say(f"\n########## subgoal {si}/{len(subgoals)}: {subgoal[:70]} ##########")
        corrections = 0
        while corrections <= MAX_CORRECTIONS:
            # local think -> directive, from graph context (THE BUS)
            ctx = g.window(subgoal + " " + task, recent_k=4, search_m=4)
            insight = _reason_directive(task, subgoal, ctx,
                                        failing if corrections else None)
            g.add_node("correction" if corrections else "directive", "reasoner", insight)
            say(f"  [reasoner] {insight[:90]}")

            # Actor: several tool steps under this directive
            progressed = False
            for step in range(STEP_BUDGET):
                total_steps += 1
                ctx = g.window(subgoal + " " + task, recent_k=4, search_m=4)
                act = _actor_action(task, subgoal, insight, ctx, failing)
                if not act:
                    g.add_node("result", "actor", "unparseable/again", payload={"passed": False})
                    continue
                tool, args, content = act
                g.add_node("action", "actor", f"{tool} {args}", payload={"tool": tool})
                result = _run_tool(tool, args, content)
                passed = tools.tests_pass()
                g.add_node("result", "actor",
                           f"{tool}: {result.splitlines()[0][:120]}", payload={"passed": passed})
                say(f"    step{step+1}: {tool} -> {'PASS' if passed else result.splitlines()[0][:50]}")
                if passed:
                    say("  verifier PASSED -> success (coupling)")
                    return True, total_steps, "coupling"
                if tool in ("write_file", "run_tests") and "FAIL" in result:
                    failing = result
                    progressed = True
                    break   # hand back to Reasoner for a correction
            corrections += 1
            if not progressed and corrections > MAX_CORRECTIONS:
                break
        # subgoal exhausted -> next subgoal (state persists in sandbox)

    if tools.tests_pass():
        return True, total_steps, "coupling"
    say("[duo2] exhausted -> single-brain net")
    return _degrade(task)
