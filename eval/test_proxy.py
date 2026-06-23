"""Test the OpenAI tool-call adapter on v12's REAL output shape (no GPU, no server).

The bridge's critical job: turn v12's text-JSON into a proper OpenAI `tool_calls` response
that OpenCode/aider parse. Run: python -m eval.test_proxy
"""
from agent.proxy import extract_tool_calls, build_response


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
