"""Task decomposition: turn one fuzzy task into an ordered list of concrete subtasks.

Why this exists: a 3B model is weak at holding a whole multi-step task in its head
(it burns the step budget thrashing between half-finished goals). It is strong at
executing ONE focused, concrete subtask. So we spend a single cheap planning turn up
front to split the work, then drive each subtask through the normal proposer loop.

Contract: the model reasons freely, then emits ONE fenced block:

    ```plan
    ["fix the off-by-one in slice()", "add the missing None guard in load()"]
    ```

A single-element list is the correct answer for an atomic task — decomposition must
never *invent* steps. We default to `[task]` on any doubt, so a planner failure
degrades to today's flat single-task behaviour rather than breaking the run.
"""
import json
import re

from . import llm

_PLAN_RE = re.compile(r"```plan\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"(?s).*</think>")
_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)

_SYSTEM = (
    "You are the planning core of an autonomous coding agent. Given a task, break it "
    "into the SMALLEST ordered list of concrete, independently-verifiable subtasks.\n\n"
    "RULES:\n"
    "- If the task is already a single atomic change, return a ONE-element list. Do NOT "
    "invent extra steps.\n"
    "- Each subtask is one short imperative sentence naming a concrete change.\n"
    "- Order them so earlier subtasks unblock later ones.\n"
    "- At most 6 subtasks.\n\n"
    "After reasoning, output EXACTLY ONE fenced block:\n"
    '```plan\n["first subtask", "second subtask"]\n```\n\n'
    "EXAMPLES:\n\n"
    "Task: Fix the off-by-one in slice_window() so the tests pass.\n"
    '```plan\n["Fix the off-by-one in slice_window()"]\n```\n\n'
    "Task: buggy.py has bugs in both to_celsius() and initials(). Fix every bug.\n"
    '```plan\n["Fix the formula bug in to_celsius()", "Fix initials() to use every word"]\n```\n\n'
    "Task: Add a /health endpoint and write a test for it.\n"
    '```plan\n["Add the /health endpoint handler", "Write a test that asserts /health returns 200"]\n```'
)


def _extract_list(text):
    """Pull the JSON string-array out of the model reply, liberally."""
    tail = _THINK_RE.sub("", text) if "</think>" in text else text
    for region in (tail, text):
        for raw in _PLAN_RE.findall(region):
            arr = _try_array(raw.strip())
            if arr is not None:
                return arr
        # fall back to the last bare [...] in the region
        for raw in reversed(_ARRAY_RE.findall(region)):
            arr = _try_array(raw)
            if arr is not None:
                return arr
    return None


def _try_array(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list) and obj and all(isinstance(x, str) for x in obj):
        cleaned = [s.strip() for s in obj if s.strip()]
        return cleaned or None
    return None


def plan(task, max_subtasks=6, temperature=0.2):
    """Return an ordered list of subtask strings. Always returns at least [task]."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": task},
    ]
    try:
        reply = llm.chat(messages, temperature=temperature, max_tokens=1024)
    except Exception as e:  # planning must never break the run
        print(f"  [planner] failed, running task whole: {e}")
        return [task]
    subtasks = _extract_list(reply)
    if not subtasks:
        return [task]
    return subtasks[:max_subtasks]
