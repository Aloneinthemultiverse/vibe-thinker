"""OpenAI-compatible proxy that presents CoupleVibe as ONE model to any harness (OpenCode,
aider, Claude-Code-style clients).

Evidence (2026-06-23) that justifies the bridge:
  - v12 (:8080) emits the RIGHT tool call but as TEXT: {"name":"read_file","arguments":{...}}
    — not in OpenAI's structured `tool_calls` field that harnesses parse.
  - base (:8082) reasons toward the intent but rambles; can't emit a clean call.
So: Reasoner THINKS -> Actor EMITS the call -> this proxy TRANSLATES v12's text-JSON into a
proper OpenAI `tool_calls` response. The harness then executes the tool and owns the workspace.

The translation (extract_tool_calls) is the critical piece and is a pure, unit-testable
function — no server, no model needed to validate it.
"""
import json
import os
import re
import time

# Match a JSON tool-call object. v12 is INCONSISTENT — it uses "name" OR "tool" as the key,
# tool names can be hyphenated (e.g. gitnexus-debugging), and args under "arguments" OR "args".
_CALL = re.compile(
    r'\{[^{}]*"(?:name|tool)"\s*:\s*"[\w.\-]+"[^{}]*"(?:arguments|args)"\s*:\s*\{[^{}]*\}[^{}]*\}',
    re.DOTALL)
# Also accept OpenAI-ish "parameters" / function wrappers and fenced json blocks.
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _coerce_args(raw):
    try:
        return json.loads(raw)
    except Exception:
        return {}


def extract_tool_calls(content):
    """Parse v12's free-text output into OpenAI `tool_calls`. Returns (tool_calls, leftover_text).
    tool_calls is [] if none found (then the content is a normal assistant message)."""
    if not content:
        return [], ""
    calls = []
    text = content
    # 1) try fenced json blocks first (cleanest)
    candidates = _FENCE.findall(content) or []
    # 2) plus inline {"name":...,"arguments":...} objects
    for m in _CALL.finditer(content):
        candidates.append(m.group(0))
    seen = set()
    for i, cand in enumerate(candidates):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments", obj.get("args", {}))
        if isinstance(args, str):
            args = _coerce_args(args)
        if not name:
            continue
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
        text = text.replace(cand, "")
    return calls, text.strip()


# Common alias mismatches between our fine-tune schema and harness tool schemas.
_ARG_ALIASES = {"path": "filePath", "file": "filePath", "filepath": "filePath",
                "file_path": "filePath", "text": "content", "code": "content"}


def _tool_names(tools):
    return [t.get("function", {}).get("name", "") for t in (tools or [])
            if t.get("function", {}).get("name")]


def _gbnf_str(s):
    """Escape a literal string for a GBNF double-quoted terminal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def tool_grammar(tools):
    """Build a GBNF grammar that forces the model to emit EXACTLY one JSON tool call
    whose `name` is one of the REAL available tools. This makes the 3B's worst failure
    mode — emitting a filename (calc.py) as the tool name — structurally impossible.
    `arguments` stays a free JSON object. Returns None if there are no named tools."""
    names = _tool_names(tools)
    if not names:
        return None
    name_alt = " | ".join(f'"\\"{_gbnf_str(n)}\\""' for n in names)
    return (
        'root    ::= "{" ws "\\"name\\"" ws ":" ws name ws "," ws '
        '"\\"arguments\\"" ws ":" ws object ws "}"\n'
        f"name    ::= {name_alt}\n"
        'object  ::= "{" ws ( pair ( ws "," ws pair )* )? ws "}"\n'
        'pair    ::= string ws ":" ws value\n'
        'value   ::= object | array | string | number | "true" | "false" | "null"\n'
        'array   ::= "[" ws ( value ( ws "," ws value )* )? ws "]"\n'
        'string  ::= "\\"" ( [^"\\\\] | "\\\\" . )* "\\""\n'
        'number  ::= "-"? [0-9]+ ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?\n'
        'ws      ::= [ \\t\\n]*\n'
    )


def _repair_name(name, args, tools):
    """Safety net for when grammar is off/unsupported and v12 emits a bad tool name.
    The signature failure is filename-as-toolname (name='calc.py') with the file body in
    args — remap to the harness write tool, moving the filename into the path arg."""
    names = _tool_names(tools)
    if not names or name in names:
        return name, args
    looks_like_file = bool(re.search(r"\.[A-Za-z0-9]{1,5}$", name)) or "/" in name or "\\" in name
    write_tool = next((n for n in ("write", "edit", "create", "create_file") if n in names), None)
    if looks_like_file and write_tool and isinstance(args, dict):
        out = dict(args)
        out.setdefault("filePath", name)
        return write_tool, out
    # Otherwise snap to a case-insensitive match if one exists; else leave as-is.
    low = {n.lower(): n for n in names}
    return low.get(name.lower(), name), args


def _relevant_tools(tools, query, k=6):
    """TOOL-KG retrieval: instead of injecting ALL tool schemas into v12 (~10k tokens),
    retrieve only the top-k tools relevant to the task. Cuts tokens drastically AND helps the
    3B pick better (fewer tools = less wandering). Built on our own VecIndex (no GPL)."""
    if not tools or len(tools) <= k:
        return tools
    from .vecindex import VecIndex, default_embedder
    vi = VecIndex(embedder=default_embedder())
    for t in tools:
        fn = t.get("function", {})
        props = (fn.get("parameters", {}) or {}).get("properties", {}) or {}
        vi.add(f"{fn.get('name','')} {fn.get('description','')} {' '.join(props)}")
    rank, _ = vi.vector_rank(query, limit=k)
    return [tools[i] for i in rank[:k]] or tools


def _last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def _required_keys(tools, name):
    for t in tools or []:
        fn = t.get("function", {})
        if fn.get("name") == name:
            params = fn.get("parameters", {}) or {}
            return params.get("required") or list((params.get("properties") or {}).keys())
    return None


def _remap_args(name, args, tools):
    """Rename v12's arg keys to the target tool's actual schema when they're obvious
    aliases (path->filePath). Only fills keys the tool wants that are missing."""
    req = _required_keys(tools, name)
    if not req or not isinstance(args, dict):
        return args
    out = dict(args)
    for k in list(out.keys()):
        if k not in req and _ARG_ALIASES.get(k) in req and _ARG_ALIASES[k] not in out:
            out[_ARG_ALIASES[k]] = out.pop(k)
    return out


def build_response(content, model="couplevibe", created=0, tools=None):
    """Wrap (possibly tool-calling) text into an OpenAI chat-completion response."""
    calls, leftover = extract_tool_calls(content)
    if tools:
        for c in calls:
            fn = c["function"]
            try:
                args = json.loads(fn["arguments"])
            except Exception:
                args = {}
            # Repair a bad/invalid tool name (e.g. filename-as-toolname) BEFORE arg remap,
            # so the harness only ever sees a real tool with the right schema keys.
            fn["name"], args = _repair_name(fn["name"], args, tools)
            fn["arguments"] = json.dumps(_remap_args(fn["name"], args, tools))
    message = {"role": "assistant", "content": leftover or (None if calls else content)}
    if calls:
        message["tool_calls"] = calls
    return {
        "id": "chatcmpl-couplevibe",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if calls else "stop"}],
    }


# --------------------------------------------------------------- live server (optional)
_ERR_MARKERS = ("not found", "error", "failed", "no such file", "invalid", "missing key")


def _is_error_result(content):
    """True if a tool RESULT looks like a failure (so it shouldn't count as progress)."""
    s = content if isinstance(content, str) else json.dumps(content or "")
    s = s.lower()
    return any(m in s for m in _ERR_MARKERS)


def _target_files(text):
    """Pull likely target filenames out of a task/message (e.g. calc.py, src/x.ts)."""
    return re.findall(r"[\w./\\-]+\.[A-Za-z][A-Za-z0-9]{0,4}\b", text or "")


def _deterministic_tool(messages, tools):
    """Reliable next-tool hint that does NOT trust the 3B for routing. The 3B's failure on
    create tasks is picking `read` on a file that doesn't exist; deterministically: if the
    task names a file that has NOT been written/read successfully yet, the next step is to
    WRITE it. Returns (tool_name, guidance) or ('', '')."""
    names = _tool_names(tools)
    write_tool = next((n for n in ("write", "create", "create_file", "edit") if n in names), None)
    if not write_tool:
        return "", ""
    task = next((m.get("content") for m in messages if m.get("role") == "user"), "") or ""
    targets = _target_files(task if isinstance(task, str) else json.dumps(task))
    if not targets:
        return "", ""
    # Has any tool already SUCCEEDED (a non-error tool result)? If so, don't force write.
    if any(m.get("role") == "tool" and not _is_error_result(m.get("content")) for m in messages):
        return "", ""
    fname = targets[0]
    # COUPLE reason->transcribe: the Reasoner (strong coder) writes the actual file content;
    # v12 just emits the write call carrying it. Fixes v12 writing prose instead of code.
    code = _reasoner_file_content(task, fname)
    if code:
        return write_tool, (f"Use the {write_tool} tool to create {fname} with EXACTLY this "
                            f"content (copy it verbatim, do not summarize):\n{code}")
    return write_tool, (f"The file {fname} does not exist yet, so create it now with the "
                        f"{write_tool} tool, writing the full required contents.")


def _reasoner_file_content(task, fname):
    """Ask the Reasoner to emit ONLY the raw file content (no prose, no fences) for a create
    task. Returns the code string, or '' on failure (caller falls back to generic guidance)."""
    from . import llm
    usr = (f"Write the COMPLETE contents of the file `{fname}` for this task:\n{task}\n\n"
           "Output ONLY the file's raw text/code. No explanation, no markdown fences.")
    try:
        r = llm.chat_reasoner([{"role": "system", "content": "You are an expert programmer."},
                               {"role": "user", "content": usr}], temperature=0.2, max_tokens=4096)
    except Exception:
        return ""
    # Strip a code fence if the model added one anyway.
    fence = re.search(r"```[a-zA-Z]*\n(.*?)```", r, re.S)
    code = fence.group(1) if fence else r
    return code.strip()


def _reasoner_judge(messages, tools):
    """THE COUPLE for a harness: the Reasoner (base 3B) looks at the task + recent tool
    results and decides DONE vs CONTINUE(+next action). Supplies the completion judgment v12
    lacks (it loops forever otherwise) and guides the next action."""
    from . import llm
    task = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    convo = "\n".join(f"{m.get('role')}: {str(m.get('content'))[:280]}" for m in messages[-6:])
    names = ", ".join(t.get("function", {}).get("name", "") for t in (tools or []))
    usr = (f"TASK: {task}\n\nRECENT ACTIVITY (newest last):\n{convo}\n\nAvailable tools: {names}\n\n"
           "Decide the SINGLE next step. If a target file does not exist yet, it must be "
           "CREATED with a write tool before it can be read. Reason briefly, then end with "
           "EXACTLY two lines:\n"
           "VERDICT: DONE        (only if the task is fully accomplished)\n"
           "  or\nVERDICT: CONTINUE - <the single next action in plain words>\n"
           "TOOL: <the exact name of the one tool to use next, from the list above>")
    try:
        r = llm.chat_reasoner([{"role": "system", "content": "You direct a coding agent's next move."},
                               {"role": "user", "content": usr}], temperature=0.3, max_tokens=1024)
    except Exception:
        return "continue", "", ""
    valid = _tool_names(tools)
    tm = re.search(r"TOOL:\s*([\w.\-]+)", r, re.I)
    tool = tm.group(1) if tm and tm.group(1) in valid else ""
    m = re.search(r"VERDICT:\s*(DONE|CONTINUE)\s*-?\s*([^\n]*)", r, re.I)
    if m and m.group(1).upper() == "DONE":
        return "done", "", ""
    return "continue", (m.group(2).strip()[:280] if m else ""), tool


def _generate(messages, tools=None):
    """Reasoner THINKS, Actor EMITS. DUO_COUPLE=1 = the couple: Reasoner judges DONE/next,
    then v12 acts on that guidance. Default (off) = proven v12-only path."""
    from . import llm
    guidance = ""
    forced_tool = ""
    if os.environ.get("DUO_COUPLE") and tools and llm.reasoner_healthy():
        # Only declare done after a tool actually SUCCEEDED — a failed read (file-not-found)
        # is still a role==tool message, so counting any tool call lets the Reasoner wrongly
        # stop before anything was accomplished. Require a non-error tool result.
        succeeded = any(m.get("role") == "tool"
                        and not _is_error_result(m.get("content")) for m in messages)
        # Only spend a Reasoner call on completion judgment AFTER something has succeeded —
        # before that, DONE can't fire anyway and deterministic routing handles the move.
        if succeeded:
            verdict, guidance, forced_tool = _reasoner_judge(messages, tools)
            if verdict == "done":
                return "Task complete."   # no tool call -> finish_reason stop -> harness stops
        # The 3B Reasoner is unreliable at routing (and may even say DONE on turn 1). For the
        # create-file case the right move is deterministic — override with it when available.
        det_tool, det_guidance = _deterministic_tool(messages, tools)
        if det_tool:
            forced_tool, guidance = det_tool, det_guidance
    sys_extra = ""
    grammar = None
    if tools:
        full_tools = tools
        # THE COUPLE STEERS THE TOOL: if a tool was forced, pick it from the FULL list FIRST —
        # before TOOL-KG narrowing, which can rank the wrong tools higher and drop the one we
        # need (e.g. drop `write` for a "build app" query, leaving only read -> wrong action).
        grammar_tools = None
        if forced_tool:
            picked = [t for t in full_tools if t.get("function", {}).get("name") == forced_tool]
            if picked:
                grammar_tools = picked
        if grammar_tools is None:
            grammar_tools = _relevant_tools(full_tools, _last_user(messages))  # TOOL-KG top-k
        tools = grammar_tools
        spec = []
        for t in grammar_tools:
            fn = t.get("function", {})
            params = fn.get("parameters", {}) or {}
            req = params.get("required") or list((params.get("properties") or {}).keys())
            spec.append(f'  {fn.get("name","")}(args: {", ".join(req) or "none"})')
        sys_extra = ("\nTo use a tool, output EXACTLY one JSON object and nothing else:\n"
                     '{"name": "<tool>", "arguments": {<exact arg keys>}}\n'
                     "Use EXACTLY these argument keys for each tool:\n" + "\n".join(spec))
        # GRAMMAR CONSTRAINT (default on; DUO_GRAMMAR=0 to disable): force the decoder to a
        # legal tool call so v12 cannot emit a filename as the tool name. When the Reasoner
        # picked a tool, this grammar admits only that one — structure AND choice enforced.
        if os.environ.get("DUO_GRAMMAR", "1") != "0":
            grammar = tool_grammar(grammar_tools)
    if guidance:
        sys_extra += f"\n\nThe reasoner advises the next action: {guidance}"
    msgs = list(messages)
    if sys_extra:
        msgs = [{"role": "system", "content": sys_extra}] + msgs
    return llm.chat(msgs, temperature=0.2, max_tokens=1024, grammar=grammar)


def _sse_chunks(resp, model):
    """Turn a finished response into OpenAI streaming SSE chunks (role -> body -> finish -> DONE).
    We generate non-streaming under the hood, then replay as a few SSE deltas, which the Vercel
    AI SDK (OpenCode) accepts."""
    msg = resp["choices"][0]["message"]
    finish = resp["choices"][0]["finish_reason"]
    base = {"id": resp["id"], "object": "chat.completion.chunk", "model": model}

    def chunk(delta, fr=None):
        return "data: " + json.dumps({**base, "choices": [
            {"index": 0, "delta": delta, "finish_reason": fr}]}) + "\n\n"

    yield chunk({"role": "assistant"})
    if msg.get("tool_calls"):
        for i, tc in enumerate(msg["tool_calls"]):
            yield chunk({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                         "function": tc["function"]}]})
    elif msg.get("content"):
        yield chunk({"content": msg["content"]})
    yield chunk({}, fr=finish)
    yield "data: [DONE]\n\n"


def serve(port=8088):  # pragma: no cover - manual/integration use
    """Start the OpenAI-compatible endpoint. Point OpenCode/aider at http://127.0.0.1:8088/v1."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                body = json.dumps({"object": "list", "data": [
                    {"id": "couplevibe", "object": "model", "owned_by": "couplevibe"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            model = body.get("model", "couplevibe")
            try:
                tools = body.get("tools")
                raw = _generate(body.get("messages", []), tools)
                resp = build_response(raw, model=model, tools=tools)
            except Exception as e:
                resp = build_response(f"error: {e}", model=model)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for c in _sse_chunks(resp, model):
                    self.wfile.write(c.encode())
                    self.wfile.flush()
            else:
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    print(f"CoupleVibe proxy on http://127.0.0.1:{port}/v1  (chat/completions + models)")
    HTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":  # pragma: no cover
    serve()
