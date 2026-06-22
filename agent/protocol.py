"""Parse and validate the model's action output.

Contract: the model reasons freely, then emits exactly one fenced block:

    ```action
    {"tool": "read_file", "args": {"path": "demo/buggy.py"}}
    ```

For write_file the new file content rides in a SEPARATE fenced python block,
so we never have to escape multiline code inside JSON:

    ```action
    {"tool": "write_file", "args": {"path": "demo/buggy.py"}}
    ```
    ```python
    <full new file content>
    ```
"""
import json
import re

TOOLS = {
    "read_file": {"path"},
    "write_file": {"path"},
    "run_tests": set(),
    "kg_query": {"query"},
    "kg_context": {"symbol"},
    "kg_impact": {"symbol"},
    "finish": set(),          # reason is informational, not required
    "respond": set(),         # not a real tool; the loop redirects it (Glaive baked it in)
}

_ACTION_RE = re.compile(r"```action\s*(.*?)```", re.DOTALL)
_PY_RE = re.compile(r"```(?:python|py)\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"(?s).*</think>")


class ParseError(Exception):
    pass


def _iter_json_objects(s):
    """Yield top-level {...} substrings (brace-balanced, string-aware)."""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield s[start:i + 1]


def _extract_action_obj(text):
    """Find the action JSON. Be liberal: VibeThinker often emits bare JSON
    after </think> instead of a fenced ```action block. Prefer post-think
    content, fall back to fenced blocks, then to any {..."tool"...} object."""
    # 1. Everything after the last </think> — that's the model's real answer.
    tail = _THINK_RE.sub("", text) if "</think>" in text else text
    for region in (tail, text):
        # fenced action blocks first (authoritative when present)
        for raw in _ACTION_RE.findall(region):
            try:
                obj = json.loads(raw.strip())
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
        # then any brace-balanced JSON object carrying a tool key
        candidates = [c for c in _iter_json_objects(region)]
        for raw in reversed(candidates):  # last one = final decision
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "tool" in obj:
                return obj
    return None


def parse_action(text):
    """Return (tool, args, content) or raise ParseError with a helpful message."""
    obj = _extract_action_obj(text)
    if obj is None:
        raise ParseError(
            "No action found. After your reasoning, output a single JSON object "
            'like {"tool": "read_file", "args": {"path": "buggy.py"}}.'
        )
    tool = obj["tool"]
    args = obj.get("args", {}) or {}
    if tool not in TOOLS:
        raise ParseError(f"Unknown tool {tool!r}. Allowed: {sorted(TOOLS)}.")
    missing = TOOLS[tool] - set(args)
    if missing:
        raise ParseError(f"Tool {tool!r} missing required args: {sorted(missing)}.")

    content = None
    if tool == "write_file":
        py = _PY_RE.findall(text)
        if py:
            content = py[-1].lstrip("\n").rstrip() + "\n"
        else:
            # Fallback: the model (Glaive-trained) often puts the file body INLINE in
            # args instead of a separate ```python block. Accept that too.
            inline = args.get("content") or args.get("code") or args.get("file")
            if isinstance(inline, str) and inline.strip():
                content = inline.lstrip("\n").rstrip() + "\n"
            else:
                raise ParseError(
                    "write_file needs the full new file content — either in a separate "
                    "```python ... ``` block, or as an \"content\" string inside args."
                )
    return tool, args, content
