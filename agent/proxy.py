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
def _generate(messages, tools=None):
    """Reasoner THINKS, Actor EMITS. Returns the Actor's raw text (to be translated)."""
    from . import llm
    # If a harness is driving with tools, let the base Reasoner add a brief plan first, then
    # have v12 emit the concrete call. (Kept simple; the adapter does the format work.)
    sys_extra = ""
    if tools:
        tools = _relevant_tools(tools, _last_user(messages))   # TOOL-KG: only relevant tools
        spec = []
        for t in tools:
            fn = t.get("function", {})
            params = fn.get("parameters", {}) or {}
            req = params.get("required") or list((params.get("properties") or {}).keys())
            spec.append(f'  {fn.get("name","")}(args: {", ".join(req) or "none"})')
        sys_extra = ("\nTo use a tool, output EXACTLY one JSON object and nothing else:\n"
                     '{"name": "<tool>", "arguments": {<exact arg keys>}}\n'
                     "Use EXACTLY these argument keys for each tool:\n" + "\n".join(spec))
    msgs = list(messages)
    if sys_extra:
        msgs = [{"role": "system", "content": sys_extra}] + msgs
    return llm.chat(msgs, temperature=0.2, max_tokens=1024)


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
