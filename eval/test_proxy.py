"""Test the OpenAI tool-call adapter on v12's REAL output shape (no GPU, no server).

The bridge's critical job: turn v12's text-JSON into a proper OpenAI `tool_calls` response
that OpenCode/aider parse. Run: python -m eval.test_proxy
"""
from agent.proxy import extract_tool_calls, build_response, tool_grammar, _repair_name

# A small realistic OpenCode-style tool set for the OpenCode-compat tests.
_TOOLS = [
    {"function": {"name": "write", "parameters": {"required": ["filePath", "content"]}}},
    {"function": {"name": "read", "parameters": {"required": ["filePath"]}}},
    {"function": {"name": "bash", "parameters": {"required": ["command"]}}},
]


def test_v12_inline_textjson():
    """v12's actual probe output: a {"name","arguments"} object in plain content."""
    content = ('Call the tool to get what\'s needed.\n'
               '{"name": "read_file","arguments": {"path": "buggy.py"}}')
    calls, leftover = extract_tool_calls(content)
    assert len(calls) == 1, f"expected 1 tool call, got {len(calls)}"
    fn = calls[0]["function"]
    assert fn["name"] == "read_file"
    import json
    assert json.loads(fn["arguments"]) == {"path": "buggy.py"}


def test_fenced_json_block():
    content = 'Sure.\n```json\n{"name": "run_tests", "arguments": {}}\n```'
    calls, _ = extract_tool_calls(content)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "run_tests"


def test_plain_text_no_tools():
    calls, leftover = extract_tool_calls("Just a normal answer, no tool needed.")
    assert calls == [] and "normal answer" in leftover


def test_build_response_openai_shape():
    content = '{"name": "read_file", "arguments": {"path": "x.py"}}'
    resp = build_response(content)
    msg = resp["choices"][0]["message"]
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["type"] == "function"
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"


def test_build_response_plain():
    resp = build_response("hello there")
    msg = resp["choices"][0]["message"]
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in msg and msg["content"] == "hello there"


def test_repair_filename_as_toolname():
    """THE OpenCode 3B ceiling failure: v12 emitted the filename 'calc.py' as the TOOL
    NAME. Repair must remap it to the write tool with the filename moved into filePath."""
    name, args = _repair_name("calc.py", {"content": "def add(a, b): return a + b"}, _TOOLS)
    assert name == "write", name
    assert args["filePath"] == "calc.py", args


def test_build_response_repairs_bad_name_end_to_end():
    """Full bridge: v12's bad text-JSON -> a VALID OpenCode write call (not 'Invalid Tool')."""
    import json
    bad = '{"name": "calc.py", "arguments": {"content": "def add(a, b): return a + b"}}'
    resp = build_response(bad, tools=_TOOLS)
    tc = resp["choices"][0]["message"]["tool_calls"][0]["function"]
    assert tc["name"] == "write", tc["name"]
    assert json.loads(tc["arguments"])["filePath"] == "calc.py"


def test_valid_name_passes_through_untouched():
    name, args = _repair_name("read", {"filePath": "x.py"}, _TOOLS)
    assert name == "read" and args == {"filePath": "x.py"}


def test_grammar_restricts_name_to_real_tools():
    g = tool_grammar(_TOOLS)
    name_rule = next(l for l in g.splitlines() if l.startswith("name"))
    assert "write" in name_rule and "read" in name_rule and "bash" in name_rule
    assert "calc" not in g  # a filename can NEVER be a legal name
    assert tool_grammar([]) is None  # no tools -> no grammar


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
