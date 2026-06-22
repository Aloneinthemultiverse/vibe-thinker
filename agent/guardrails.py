"""Step 7 — guardrails: the orchestrator decides what the brain is ALLOWED to do.

The model only PROPOSES actions; this layer is the gate that runs before any tool
executes. It enforces four things, all fail-safe (deny on doubt):

  1. Risk policy   — every tool has a risk level; dangerous ones (deploy/delete/send)
                     require explicit approval. In autonomous mode, default = DENY.
  2. Permissions   — an allow/deny list; unknown tools are denied, not run.
  3. Loop detection— the same (tool, args) repeated too often is a stuck loop; block it.
  4. Budgets       — a hard cap on total tool calls per run.

A blocked action never throws into the tool; it returns (False, reason) so the loop
can feed the reason back to the model and let it choose differently.
"""
import json
from collections import deque

# Risk levels. Anything not listed is treated as UNKNOWN -> denied.
RISK = {
    "read_file": "safe",
    "kg_query": "safe",
    "kg_context": "safe",
    "kg_impact": "safe",
    "run_tests": "safe",      # sandboxed verifier
    "write_file": "write",    # mutates sandbox files
    "finish": "safe",
    # dangerous, gated by default:
    "delete_file": "dangerous",
    "deploy": "dangerous",
    "send": "dangerous",
    "shell": "dangerous",
}

SAFE_AUTO = {"safe", "write"}     # auto-allowed risk levels in autonomous mode


class GuardError(Exception):
    pass


class Guard:
    def __init__(self, approve=None, max_calls=20, loop_window=6,
                 loop_repeats=3, allow=None, deny=None):
        # approve: callable(tool, args)->bool for dangerous actions. None = autonomous.
        self.approve = approve
        self.max_calls = max_calls
        self.calls = 0
        self.history = deque(maxlen=loop_window)
        self.loop_repeats = loop_repeats
        self.allow = set(allow) if allow else None    # if set, ONLY these allowed
        self.deny = set(deny) if deny else set()

    def _sig(self, tool, args):
        return tool + ":" + json.dumps(args or {}, sort_keys=True)

    def check(self, tool, args):
        # budget first — a hard ceiling.
        if self.calls >= self.max_calls:
            return False, f"tool-call budget exhausted ({self.max_calls})."
        # explicit deny / allow-list.
        if tool in self.deny:
            return False, f"tool {tool!r} is on the deny list."
        if self.allow is not None and tool not in self.allow:
            return False, f"tool {tool!r} not in the allow list."
        # unknown tool -> fail safe.
        risk = RISK.get(tool)
        if risk is None:
            return False, f"unknown tool {tool!r}; denied (fail-safe)."
        # loop detection.
        sig = self._sig(tool, args)
        if sum(1 for s in self.history if s == sig) >= self.loop_repeats - 1:
            return False, (f"loop detected: {tool} with the same args "
                           f"{self.loop_repeats}x. Try a different action.")
        # risk gate.
        if risk not in SAFE_AUTO:
            if self.approve is None:
                return False, (f"action {tool!r} is dangerous and requires approval; "
                               "none available in autonomous mode. Denied.")
            if not self.approve(tool, args):
                return False, f"approver rejected dangerous action {tool!r}."
        return True, "ok"

    def commit(self, tool, args):
        """Call AFTER an allowed action actually ran, to advance counters."""
        self.calls += 1
        self.history.append(self._sig(tool, args))


def _selftest():
    g = Guard(max_calls=5)
    # safe allowed
    assert g.check("read_file", {"path": "x"})[0] is True
    # unknown denied
    ok, why = g.check("rm_rf", {})
    assert not ok and "unknown" in why
    # dangerous denied in autonomous mode
    ok, why = g.check("deploy", {"env": "prod"})
    assert not ok and "dangerous" in why
    # dangerous allowed with an approver that says yes
    g2 = Guard(approve=lambda t, a: True)
    assert g2.check("deploy", {"env": "prod"})[0] is True
    # loop detection: same action 3x
    g3 = Guard()
    a = {"path": "f"}
    assert g3.check("write_file", a)[0] is True
    g3.commit("write_file", a)
    g3.commit("write_file", a)
    ok, why = g3.check("write_file", a)
    assert not ok and "loop" in why, why
    # budget enforcement
    g4 = Guard(max_calls=2)
    for _ in range(2):
        assert g4.check("read_file", {})[0]
        g4.commit("read_file", {})
    ok, why = g4.check("read_file", {})
    assert not ok and "budget" in why
    # deny list
    g5 = Guard(deny={"write_file"})
    assert g5.check("write_file", {"path": "x"})[0] is False
    print("OK: risk gate, unknown-deny, approval, loop detection, budget, deny-list verified.")


if __name__ == "__main__":
    _selftest()
