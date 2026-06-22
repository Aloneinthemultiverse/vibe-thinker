"""Step 8 — skills + the constitution.

CONSTITUTION: a versioned, disciplined system prompt — role, strict output format,
propose-never-execute, safety boundaries. One source of truth, version-stamped so we
can ablate prompt changes.

SKILLS: frozen, proven procedures the 3B should NOT re-derive every time (it's a small
model — re-deriving wastes steps and invites mistakes). The orchestrator retrieves the
2-3 skills relevant to the task by keyword and injects only those, keeping the prompt
tight. Skills are earned: you add one after you watch the model fumble the same thing twice.
"""
import re

CONSTITUTION_VERSION = "1.0.0"

CONSTITUTION = f"""\
You are the reasoning core of an autonomous coding agent (constitution v{CONSTITUTION_VERSION}).

ROLE: You PROPOSE one action at a time. You NEVER execute anything yourself — a
deterministic orchestrator runs your proposed action and returns the result.

OUTPUT FORMAT (every turn): reason briefly, then emit EXACTLY ONE action as a JSON
object, e.g. {{"tool": "read_file", "args": {{"path": "buggy.py"}}}}. For write_file,
follow the action with the FULL new file content in a separate ```python block.

DISCIPLINE:
- One action per turn. Inspect before you edit. Always run_tests after a write.
- Do not repeat an identical action; if blocked, change your approach.
- Only finish when the verifier (tests) PASSES.

SAFETY: dangerous actions (delete, deploy, send, shell) require human approval and are
denied by default. Never attempt to bypass the guardrails."""


class Skill:
    def __init__(self, name, triggers, steps):
        self.name = name
        self.triggers = [re.compile(t, re.I) for t in triggers]
        self.steps = steps

    def matches(self, task):
        return any(t.search(task) for t in self.triggers)

    def render(self):
        return f"SKILL [{self.name}]:\n{self.steps}"


# Frozen procedures. Each was/would-be earned by an observed failure mode.
SKILLS = [
    Skill(
        "fix-failing-test",
        [r"\btest", r"\bbug", r"\bfail", r"\bfix"],
        "1) read_file the target. 2) run_tests to see the exact failure. 3) write_file "
        "the minimal corrected version. 4) run_tests again. 5) finish only on PASS."),
    Skill(
        "understand-before-change",
        [r"\bchange", r"\brefactor", r"\bimpact", r"\bdepend", r"\bblast"],
        "Before editing across files: kg_context the symbol to see callers/callees, then "
        "kg_impact to learn the blast radius. Edit only after you know what depends on it."),
    Skill(
        "locate-by-meaning",
        [r"\bfind", r"\bwhere", r"\bsearch", r"\bwhich file", r"\blocate"],
        "To find code by concept (not exact name): kg_query with a natural-language "
        "description; it returns the execution flows and files most related to the idea."),
]


def retrieve_skills(task, limit=2):
    hits = [s for s in SKILLS if s.matches(task)]
    return hits[:limit]


def build_system_prompt(task, tool_help):
    """Constitution + the tool catalogue + only the skills relevant to THIS task."""
    parts = [CONSTITUTION, "", tool_help]
    skills = retrieve_skills(task)
    if skills:
        parts.append("\nRelevant proven procedures:")
        parts.extend(s.render() for s in skills)
    return "\n".join(parts), [s.name for s in skills]


def _selftest():
    p, used = build_system_prompt("fix the failing merge_intervals test", "TOOLS...")
    assert "fix-failing-test" in used, used
    assert f"v{CONSTITUTION_VERSION}" in p
    p2, used2 = build_system_prompt("what is the blast radius of changing Bench?", "T")
    assert "understand-before-change" in used2, used2
    p3, used3 = build_system_prompt("find where benchmarks are run", "T")
    assert "locate-by-meaning" in used3, used3
    # unrelated task -> no skills injected (prompt stays tight)
    _, used4 = build_system_prompt("say hello", "T")
    assert used4 == [], used4
    print(f"OK: constitution v{CONSTITUTION_VERSION} + skill retrieval verified "
          f"(3 skills, correct routing, no-match stays empty).")


if __name__ == "__main__":
    _selftest()
