"""The minimal orchestrator: brain proposes -> code runs one tool -> verify -> repeat.

The model NEVER executes anything. It only proposes an action in a strict format.
The orchestrator parses, validates, runs the tool, and feeds the result back.
"""
import json
import os
import time

from . import llm, tools
from .guardrails import Guard
from .memory import Memory
from .protocol import parse_action, ParseError
from .skills import build_system_prompt

TOOL_HELP = """Available tools:
- read_file   args: {"path": "<file>"}            -> returns file contents
- write_file  args: {"path": "<file>"}            -> writes a file. You MUST also
                 include the FULL new file content in a separate ```python block.
- run_tests   args: {}                            -> runs the test suite (the verifier)
- kg_query    args: {"query": "<concept>"}        -> search the code knowledge graph
                 for execution flows / relationships related to a concept
- kg_context  args: {"symbol": "<name>"}          -> 360 view of a symbol: callers, callees
- kg_impact   args: {"symbol": "<name>"}          -> blast radius: what breaks if changed
- finish      args: {"reason": "<why>"}           -> stop; only after tests PASS

The kg_* tools read an EXACT knowledge graph of the wider codebase (not the
sandbox file). Use them to understand structure before editing across files.

For write_file, follow the action JSON with the full file in a ```python block."""

# Trimmed catalogue for self-contained sandbox tasks where the wider-codebase KG is
# irrelevant noise (integrated-test finding: exposing kg_* on an unindexed sandbox
# made the model wander into them instead of fixing the file).
TOOL_HELP_NO_KG = """Available tools:
- read_file   args: {"path": "<file>"}            -> returns file contents
- write_file  args: {"path": "<file>"}            -> writes a file. You MUST also
                 include the FULL new file content in a separate ```python block.
- run_tests   args: {}                            -> runs the test suite (the verifier)
- finish      args: {"reason": "<why>"}           -> stop; only after tests PASS

For write_file, follow the action JSON with the full file in a ```python block."""

KG_TOOLS = {"kg_query", "kg_context", "kg_impact"}

TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "runtime", "traces")

# How many consecutive failed verifies before the harness stops nudging the 3B and
# escalates the stuck step to a bigger model (or best-of-N). Tunable via env.
ESCALATE_AFTER = int(os.environ.get("ESCALATE_AFTER", "2"))
BEST_OF_N = int(os.environ.get("ESCALATE_BEST_OF_N", "3"))

# Think-before-act: VibeThinker is a REASONING model, but our action protocol trained it
# to "reason briefly". This restores its strength — a deep reasoning pass EACH step before
# the action. Tokens are free locally. Reasoning is EPHEMERAL (informs the action but is
# NOT kept in history) so context doesn't balloon across steps. Toggle via THINK_OFF.
THINK_TOKENS = int(os.environ.get("THINK_TOKENS", "4096"))
# Single-call think-then-act: keeps the learned "action JSON + ```python block together"
# pattern intact (the two-call split broke it), while nudging a reasoning pass first. The
# parser extracts the action + code from anywhere in the reply, so long reasoning is fine.
THINK_ACT_PROMPT = (
    "Work in TWO parts in your reply:\n"
    "1) THINK: reason step by step about the current state and the bug. If writing code, "
    "work out the EXACT corrected code and mentally check it against the failing test.\n"
    "2) ACT: then emit EXACTLY ONE action as a JSON object in the required format. If it is "
    "write_file, you MUST include the FULL corrected file in a ```python block.")


def _decide(messages, temperature, think):
    """Produce the next action reply. think=True nudges a reasoning pass in the SAME reply
    (high token budget — free locally) before the action; the parser handles both."""
    if not think:
        return llm.chat(messages, temperature=temperature)
    reply = llm.chat(messages + [{"role": "user", "content": THINK_ACT_PROMPT}],
                     temperature=temperature, max_tokens=THINK_TOKENS)
    print(f"  [think] {len(reply)} chars")
    return reply


def _hint_from_tests(test_output):
    """Turn raw verifier output into a SHORT, actionable nudge (option 2:
    retry-with-hint). Small models fix far more when shown exactly what broke and
    told to compare expected vs actual, instead of just 'tests failed'."""
    if not test_output or "FAIL" not in test_output:
        return ""
    lines = [l.strip() for l in test_output.splitlines() if l.strip()]
    assertion = next((l for l in reversed(lines)
                      if "Error" in l or l.startswith("assert") or "!=" in l
                      or "==" in l), "")
    failed_test = next((l for l in lines if l.startswith(("FAIL:", "ERROR:"))
                        or "def test" in l), "")
    bits = []
    if failed_test:
        bits.append(f"Failing test: {failed_test[:160]}")
    if assertion:
        bits.append(f"The error was: {assertion[:200]}")
    bits.append("Compare the EXPECTED value vs what your code ACTUALLY returns, find "
                "the single line responsible, and rewrite the FULL corrected file. "
                "Do not repeat your previous attempt verbatim.")
    return "\n".join(bits)


def _escalate(task, file_path, current_src, test_output, log):
    """Stuck-step rescue. Produce a corrected file body for `file_path`.

    Strategy: if a bigger model is configured, ask IT (it has the reasoning the 3B
    lacks). Otherwise fall back to best-of-N on the local model at higher
    temperature — more samples, keep whichever the verifier accepts. Returns the
    new file content string, or None if nothing helped.
    """
    # think-style recall AT THE ESCALATION POINT: now the actual file + failing test
    # are known (real function names, real error), so retrieval is well-targeted — far
    # better than recalling off the generic top-level task string. (gbrain `think`.)
    recall = ""
    mem = _episodic()
    if mem:
        try:
            from .episodic import synthesize_recall
            q = f"{current_src}\n{test_output[:400]}"
            hits = mem.search(q, k=2)
            if hits:
                recall = "\n\n" + synthesize_recall(hits)
                print(f"  [escalate] brain recall (fused {hits[0]['score']:.3f})")
        except Exception:
            pass

    prompt = [
        {"role": "system", "content":
            "You are a senior engineer fixing a failing Python file. Reply with ONLY "
            "the complete corrected file inside a single ```python code block — no prose."},
        {"role": "user", "content":
            f"TASK: {task}\n\nCURRENT FILE ({file_path}):\n```python\n{current_src}\n```\n\n"
            f"FAILING TEST OUTPUT:\n{test_output[:1500]}{recall}\n\n"
            "Return the full corrected file."},
    ]

    def _extract(reply):
        import re
        m = re.findall(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
        body = (m[-1] if m else reply).strip()
        return body + "\n" if body and "def " in body else None

    if llm.big_available():
        print("  [escalate] handing stuck step to BIG model...")
        log({"event": "escalate", "mode": "big", "file": file_path})
        try:
            cand = _extract(llm.chat_big(prompt, temperature=0.2))
            if cand:
                ok = _try_candidate(file_path, cand)
                log({"event": "escalate_result", "mode": "big", "passed": ok})
                if ok:
                    return cand
        except Exception as e:
            print(f"  [escalate] big model failed: {e}")
            log({"event": "escalate_error", "mode": "big", "error": str(e)})

    # Fallback: best-of-N on the local 3B (no big model, or it didn't pass).
    print(f"  [escalate] best-of-{BEST_OF_N} on local model...")
    log({"event": "escalate", "mode": f"best_of_{BEST_OF_N}", "file": file_path})
    for i in range(BEST_OF_N):
        try:
            cand = _extract(llm.chat(prompt, temperature=0.7 + 0.1 * i))
        except Exception:
            continue
        if cand and _try_candidate(file_path, cand):
            log({"event": "escalate_result", "mode": "best_of_n", "passed": True,
                 "sample": i})
            return cand
    log({"event": "escalate_result", "passed": False})
    return None


def _try_candidate(rel_path, content):
    """Write a candidate fix and report whether the verifier now passes. Restores
    the prior file content on failure so a bad escalation never corrupts state."""
    full = tools._safe_path(rel_path)
    prev = None
    if os.path.exists(full):
        with open(full, "r", encoding="utf-8") as f:
            prev = f.read()
    try:
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        if tools.tests_pass():
            return True
    except Exception:
        pass
    if prev is not None:
        with open(full, "w", encoding="utf-8") as f:
            f.write(prev)
    return False


def _persist(task, outcome, steps):
    """Write path (Step 6): a finished task records what it learned."""
    try:
        m = Memory()
        m.record(f"task: {task[:200]} -> {outcome} in {steps} step(s)",
                 kind="outcome")
        m.promote()
        m.close()
    except Exception as e:  # memory must never break the loop
        print(f"  [memory] write-back skipped: {e}")


_EPISODIC = None


def _episodic():
    """Lazily build the episodic memory; None if unavailable (never breaks run)."""
    global _EPISODIC
    if _EPISODIC is None:
        if os.environ.get("EPISODIC_OFF"):
            return None
        try:
            from .episodic import EpisodicMemory
            _EPISODIC = EpisodicMemory()
        except Exception as e:
            print(f"  [episodic] disabled: {e}")
            _EPISODIC = False
    return _EPISODIC or None


def _remember(task, rel_path):
    """Store a solved episode (task -> corrected file) for future recall."""
    mem = _episodic()
    if not mem:
        return
    try:
        sol = tools.read_file({"path": rel_path})
        if sol and not sol.startswith("ERROR"):
            mem.add(task, sol)
    except Exception as e:
        print(f"  [episodic] store skipped: {e}")


def run(task, max_steps=8, require_tests=True, guard=None,
        temperature=0.2, expose_kg=True, use_episodic=True, think=None):
    if think is None:
        # OFF by default: measured 2026-06-22 that prompt-forced reasoning REGRESSES v12
        # (brevity is baked into the weights -> high variance + broken action format).
        # Opt in with THINK_ON for experiments. The proven harness path is think=False.
        think = bool(os.environ.get("THINK_ON"))
    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = os.path.join(TRACE_DIR, f"trace-{int(time.time())}.jsonl")
    if guard is None:
        deny = set() if expose_kg else KG_TOOLS
        guard = Guard(max_calls=max_steps * 3, deny=deny)
    tool_help = TOOL_HELP if expose_kg else TOOL_HELP_NO_KG
    sys_prompt, skills_used = build_system_prompt(task, tool_help)
    user_msg = task
    # --- brain: recall similar solved episodes and put them in context ----------
    if use_episodic:
        mem = _episodic()
        if mem:
            try:
                from .episodic import synthesize_recall
                hits = mem.search(task, k=2)
                if hits:
                    block = synthesize_recall(hits)
                    top = hits[0]
                    print(f"  [episodic] recalled {len(hits)} (fused {top['score']:.3f}"
                          f" | vec {top['vec']:.2f} bm25 {top['bm25']:.1f}"
                          f" graph {top['graph']})")
                    user_msg = f"{task}\n\n{block}"
            except Exception as e:
                print(f"  [episodic] recall skipped: {e}")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]

    def log(event):
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    log({"event": "start", "task": task, "max_steps": max_steps,
         "skills": skills_used})
    print(f"[trace] {trace_path}")

    fail_streak = 0          # consecutive non-progress steps -> drives escalation
    last_write_path = None   # most recent file the model wrote (escalation target)
    last_read_path = None    # most recent file the model read (fallback target)
    escalated = [False]      # escalate at most once per run()

    def try_escalate(step):
        """If stuck and a target file is known, hand it to a bigger model / best-of-N.
        Returns (ok, step) to return from run(), or None to keep looping."""
        if escalated[0] or fail_streak < ESCALATE_AFTER:
            return None
        target = last_write_path or last_read_path
        if not target:
            return None
        escalated[0] = True
        try:
            src = tools.read_file({"path": target})
        except Exception:
            src = ""
        test_out = tools.run_tests()
        fix = _escalate(task, target, src, test_out, log)
        if fix is not None and tools.tests_pass():
            print("  [escalate] verifier PASSED -> success")
            log({"event": "success", "step": step, "via": "escalation"})
            _persist(task, "SUCCESS_ESCALATED", step)
            if use_episodic:
                _remember(task, target)  # learn from the big model's fix
            return True, step
        return None

    for step in range(1, max_steps + 1):
        print(f"\n=== Step {step}/{max_steps} ===")
        try:
            reply = _decide(messages, temperature, think)
        except Exception as e:
            # An LLM/HTTP error (even after trim) must never crash the run.
            print(f"  [llm error, fail-safe] {e}")
            log({"event": "llm_error", "step": step, "error": str(e)})
            fail_streak += 1
            esc = try_escalate(step)
            if esc:
                return esc
            continue

        try:
            tool, args, content = parse_action(reply)
        except ParseError as e:
            print(f"  parse failed (fail-safe): {e}")
            log({"event": "parse_error", "step": step, "reply": reply, "error": str(e)})
            # Keep only the tail of the reply in history (the long reasoning is ephemeral
            # — storing it whole would balloon context and cause ctx-overflow 400s).
            messages.append({"role": "assistant", "content": reply[-400:]})
            messages.append({"role": "user",
                             "content": f"Your last action could not be parsed: {e} Try again."})
            fail_streak += 1
            esc = try_escalate(step)
            if esc:
                return esc
            continue

        # Store ONLY the compact action in history, not the verbose reasoning — keeps the
        # context lean across steps so deep per-step reasoning never overflows n_ctx.
        _compact = json.dumps({"tool": tool, "args": args})
        if content:
            _compact += "\n```python\n" + content + "```"
        messages.append({"role": "assistant", "content": _compact})

        print(f"  action: {tool} {args}")
        log({"event": "action", "step": step, "tool": tool, "args": args,
             "reply": reply, "content": content})

        allowed, why = guard.check(tool, args)
        if not allowed:
            print(f"  BLOCKED by guardrails: {why}")
            log({"event": "blocked", "step": step, "tool": tool, "reason": why})
            messages.append({"role": "user",
                             "content": f"Action blocked by guardrails: {why}"})
            # A guardrail loop-block IS a stuck signal — count it and maybe escalate.
            fail_streak += 1
            esc = try_escalate(step)
            if esc:
                return esc
            continue
        guard.commit(tool, args)

        if tool == "respond":
            # Not a real tool — v11 inherited it from Glaive and uses it to "talk".
            # Redirect hard: keep the model's reasoning, demand a real action next.
            said = (args.get("text") or args.get("message") or "").strip()
            print(f"  respond (redirected): {said[:80]}")
            log({"event": "respond_redirect", "step": step, "text": said})
            messages.append({"role": "user", "content": (
                "There is no respond tool, so nothing happened. To make progress emit a "
                "REAL action now: read_file, run_tests, or write_file (with the full file "
                'content inline as args.content). Example: {"tool": "write_file", "args": '
                '{"path": "buggy.py", "content": "def f():\\n    return 1\\n"}}')})
            continue

        if tool == "finish":
            ok = tools.tests_pass() if require_tests else True
            print(f"  finish requested. final verifier: {'PASS' if ok else 'FAIL'}")
            log({"event": "finish", "step": step, "verified": ok})
            _persist(task, "SUCCESS" if ok else "FINISH_UNVERIFIED", step)
            return ok, step

        # A tool raising must NEVER crash the orchestrator — fail safe and feed the
        # error back (integrated-test finding 2026-06-21: a sandbox-escape ValueError
        # from a model-supplied bad path killed the whole run).
        try:
            if tool == "read_file":
                result = tools.read_file(args)
            elif tool == "write_file":
                # Verification is the center: auto-run the verifier after every write
                # so the model can never fly blind on an unverified edit.
                result = tools.write_file(args, content)
                result += "\n" + tools.run_tests()
            elif tool == "run_tests":
                result = tools.run_tests()
            elif tool == "kg_query":
                result = tools.kg_query(args)
            elif tool == "kg_context":
                result = tools.kg_context(args)
            elif tool == "kg_impact":
                result = tools.kg_impact(args)
            else:
                result = f"ERROR: unhandled tool {tool}"
        except Exception as e:
            result = f"ERROR: tool {tool} failed: {e}"

        print(f"  result: {result.splitlines()[0] if result else ''}")
        log({"event": "result", "step": step, "tool": tool, "result": result})

        if tool == "write_file":
            last_write_path = args.get("path", last_write_path)
        elif tool == "read_file" and not str(result).startswith("ERROR"):
            last_read_path = args.get("path", last_read_path)

        # Deterministic early exit: if the verifier passes, we're done.
        # (write_file now auto-verifies, so a correct fix succeeds immediately.)
        if tool in ("run_tests", "write_file") and tools.tests_pass():
            print("  verifier PASSED -> success")
            log({"event": "success", "step": step})
            _persist(task, "SUCCESS", step)
            if use_episodic and (last_write_path or last_read_path):
                _remember(task, last_write_path or last_read_path)
            return True, step

        # --- option 2: retry-with-hint -------------------------------------------
        # A verify that just FAILED: augment the bare output with an actionable hint
        # so the small model knows precisely what to fix, not just "tests failed".
        feedback = f"Tool result:\n{result}"
        if tool in ("run_tests", "write_file") and "FAIL" in result:
            fail_streak += 1
            hint = _hint_from_tests(result)
            if hint:
                feedback += f"\n\nHINT (attempt {fail_streak}):\n{hint}"
        elif tool in ("read_file", "kg_query", "kg_context", "kg_impact"):
            pass  # neutral steps don't reset the streak
        else:
            fail_streak = 0
        messages.append({"role": "user", "content": feedback})

        # --- option 1: escalation -------------------------------------------------
        # The 3B is stuck (N non-progress steps). Hand the stuck file to a bigger
        # model (or best-of-N locally). If it lands a passing fix, we're done.
        esc = try_escalate(step)
        if esc:
            return esc

    log({"event": "budget_exhausted"})
    _persist(task, "FAILURE_BUDGET", max_steps)
    return False, max_steps


def run_planned(task, max_steps=8, require_tests=True, temperature=0.2,
                expose_kg=True, max_subtasks=6):
    """Decompose `task` into subtasks, then run each through `run()` in ORDER,
    sharing one sandbox so earlier edits persist for later subtasks.

    Atomic tasks decompose to a single subtask, so this collapses to exactly one
    `run()` call — i.e. no behaviour change and no extra cost for simple work. The
    win is on multi-step tasks: the model attacks one concrete goal at a time
    instead of thrashing across the whole job within a single step budget.

    Returns (overall_ok, total_steps). overall_ok is the FINAL verifier verdict,
    so a plan that fixes everything but mis-orders a step still reports honestly.
    """
    from .planner import plan

    subtasks = plan(task, max_subtasks=max_subtasks, temperature=temperature)
    print(f"[plan] {len(subtasks)} subtask(s):")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. {st}")

    if len(subtasks) == 1:
        return run(subtasks[0], max_steps=max_steps, require_tests=require_tests,
                   temperature=temperature, expose_kg=expose_kg)

    total_steps = 0
    ok = False
    for i, sub in enumerate(subtasks, 1):
        print(f"\n########## Subtask {i}/{len(subtasks)}: {sub} ##########")
        # Each subtask is framed with the overall goal so the model keeps context,
        # but is told to focus on just this step.
        framed = (f"OVERALL GOAL: {task}\n\nFOCUS ON THIS STEP ONLY: {sub}")
        sub_ok, steps = run(framed, max_steps=max_steps, require_tests=False,
                            temperature=temperature, expose_kg=expose_kg)
        total_steps += steps
        # Deterministic cross-subtask exit: if the whole suite already passes,
        # later subtasks are redundant — stop early.
        if tools.tests_pass():
            print(f"  [plan] overall verifier PASSED after subtask {i} -> done")
            ok = True
            break
    if not ok and require_tests:
        ok = tools.tests_pass()
    _persist(task, "SUCCESS" if ok else "FAILURE_PLAN", total_steps)
    return ok, total_steps
