"""Thin client for a local llama-server (OpenAI-compatible /v1/chat/completions)."""
import json
import os
import urllib.request
import urllib.error

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

# Dual-brain Reasoner: the un-fine-tuned BASE VibeThinker-3B served in chat mode on its
# own port (long chain-of-thought; v12 on :8080 is the Actor). Env-overridable so a single
# shared server can be reused if needed.
REASONER_ENDPOINT = os.environ.get(
    "REASONER_ENDPOINT", "http://127.0.0.1:8082/v1/chat/completions")

# Escalation target: a BIGGER model used only when the 3B is stuck. Pluggable via
# env so the harness stays zero-config — if unset, the loop falls back to best-of-N
# on the local model (see loop.escalate). OpenAI-compatible endpoint + optional key.
BIG_ENDPOINT = os.environ.get("BIG_MODEL_ENDPOINT")        # e.g. https://api.../v1/chat/completions
BIG_MODEL = os.environ.get("BIG_MODEL_NAME", "")            # e.g. claude-... / gpt-...
BIG_KEY = os.environ.get("BIG_MODEL_KEY", "")


def _post(endpoint, body, headers, timeout):
    req = urllib.request.Request(endpoint, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _trim(messages, keep_tail=8):
    """Keep system + first user turn (the task) + the most recent exchanges. Used to
    recover from a context-overflow 400 on long stuck loops without losing the goal."""
    if len(messages) <= keep_tail + 2:
        return messages
    head = messages[:2]
    tail = messages[-keep_tail:]
    return head + tail


def _body(messages, temperature, top_p, max_tokens, grammar=None):
    payload = {
        "messages": messages, "temperature": temperature, "top_p": top_p,
        "max_tokens": max_tokens, "stream": False,
    }
    # GBNF grammar (llama-server) constrains the decoder to a legal output shape — used by
    # the proxy to force v12 to emit a tool call whose `name` is a REAL tool (no more
    # filename-as-toolname). Ignored by servers that don't support it.
    if grammar:
        payload["grammar"] = grammar
    return json.dumps(payload).encode("utf-8")


def chat(messages, temperature=0.6, top_p=0.95, max_tokens=2048, grammar=None):
    hdr = {"Content-Type": "application/json"}
    try:
        return _post(ENDPOINT, _body(messages, temperature, top_p, max_tokens, grammar),
                     hdr, timeout=300)
    except urllib.error.HTTPError as e:
        # 400 on a long conversation = prompt exceeded n_ctx. Retry once on a trimmed
        # history (keep the task + recent turns) instead of dying the whole run.
        if e.code == 400 and len(messages) > 10:
            trimmed = _trim(messages)
            return _post(ENDPOINT, _body(trimmed, temperature, top_p, max_tokens, grammar),
                         hdr, timeout=300)
        raise


def big_available():
    """True if a bigger escalation model is configured."""
    return bool(BIG_ENDPOINT)


def chat_big(messages, temperature=0.2, max_tokens=2048):
    """Call the configured BIGGER model. Raises if none is configured — callers
    that want a graceful fallback should check big_available() first."""
    if not BIG_ENDPOINT:
        raise RuntimeError("no BIG_MODEL_ENDPOINT configured")
    payload = {"messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    if BIG_MODEL:
        payload["model"] = BIG_MODEL
    headers = {"Content-Type": "application/json"}
    if BIG_KEY:
        headers["Authorization"] = f"Bearer {BIG_KEY}"
    return _post(BIG_ENDPOINT, json.dumps(payload).encode("utf-8"), headers, timeout=300)


def chat_reasoner(messages, temperature=0.6, top_p=0.95, max_tokens=4096):
    """Call the BASE-model Reasoner (long CoT). Higher default max_tokens than the Actor
    because we WANT long reasoning here — the base model's strength v12 lost. Same 400
    trim-and-retry recovery as chat()."""
    hdr = {"Content-Type": "application/json"}
    try:
        return _post(REASONER_ENDPOINT, _body(messages, temperature, top_p, max_tokens),
                     hdr, timeout=300)
    except urllib.error.HTTPError as e:
        if e.code == 400 and len(messages) > 10:
            trimmed = _trim(messages)
            return _post(REASONER_ENDPOINT, _body(trimmed, temperature, top_p, max_tokens),
                         hdr, timeout=300)
        raise


def _health(base_url, timeout=2):
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def healthy():
    return _health("http://127.0.0.1:8080")


def reasoner_healthy():
    """True if the Reasoner server is up (derive base URL from REASONER_ENDPOINT)."""
    base = REASONER_ENDPOINT.split("/v1/")[0]
    return _health(base)
