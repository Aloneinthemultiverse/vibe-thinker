# CoupleVibe LLM — Dual-Brain (Coupled) Architecture
> Formerly "VibeThinker-OS". Branded **CoupleVibe LLM** — two coupled 3B brains (Reasoner + Actor).

**Status:** DESIGN (not built). Author handoff doc. 2026-06-22.
**One line:** Two VibeThinker models — a *Reasoner* and an *Actor* — coupled through a
shared knowledge graph, presented externally as **one** agent. No fine-tuning, no external
model, fully local ($0).

---

## 1. Why this exists

We fine-tuned VibeThinker into **v12** to make it tool-callable. That SFT (every example
said *"reason briefly, then emit ONE action"*) accidentally **suppressed the long
chain-of-thought** the base model is good at. Measured: prompt-forcing reasoning back into
v12 regresses it (high variance, broken action format) — *a prompt cannot undo training.*

So one 3B can be a strong **reasoner** OR a reliable **actor**, not both. Instead of
retraining, **couple both models** and let each do only what it is good at:

| Brain | Model | Strength | Never does |
|---|---|---|---|
| **Reasoner** | base `VibeThinker-3B` (un-fine-tuned) | long CoT, hard logic | emit tool calls |
| **Actor** | `v12` (tool-tuned) | clean tool calls + code | invent the hard insight |

The verifier (run tests) stays the single source of truth — it anchors the whole system so
the two brains can never run away.

---

## 2. The blackboard pattern (how they communicate)

The two brains **do not call each other directly.** They read/write a **shared graph**
(the "blackboard"). This decouples them — either brain can be swapped or fixed without
touching the other (directly addresses the "fix one thing, break another" risk).

```
        ┌──────────────────── ONE AGENT (façade: agent.solve(task)) ───────────────────┐
        │                                                                                │
  ┌─────▼───────────────┐                                          ┌─────────────────────▼─┐
  │ REASONER (base 3B)  │                                          │ ACTOR (v12)            │
  │  + private graph    │                                          │  + private graph       │
  │  (hypotheses,       │                                          │  (actions, results,    │
  │   reasoning traces) │                                          │   what worked/failed)  │
  └─────────┬───────────┘                                          └───────────┬────────────┘
            │  write INSIGHT                                          write OUTCOME│
            │  read OUTCOME                                            read INSIGHT │
            └───────────────────────►  SHARED GRAPH (blackboard)  ◄───────────────┘
                                       nodes: insight/action/result/fact
                                       edges: led_to / refuted / supports / about
                                                     │
                                              ┌──────▼──────┐
                                              │  VERIFIER   │  tests pass → STOP
                                              └─────────────┘
```

### Control loop — Reasoner as active monitor/director ("the couple")

Think-before-act is **structural here, not a prompt.** Every Actor turn is preceded by a
Reasoner turn, so the system *always* thinks before it acts — without forcing the regressing
"reason briefly" prompt into v12 (whose weights suppress long CoT; proven, see §1). The
Reasoner does the thinking; the Actor does the acting; neither is asked to do the other's job.

Thinking happens at **two tiers** — global once, then local every step:

- **Tier G — global think-before-act (once, at task start).** The Reasoner does its full
  long-CoT over the *whole* problem: decompose it, enumerate every bug/subgoal, decide the
  overall order + strategy. Output = a **`plan`** node on the shared graph (an ordered list of
  subgoals). This is Plan-and-Execute, and it's where the base model's long reasoning — the
  thing v12 lost — actually earns its place. Done ONCE; refreshed only if the plan is proven
  wrong (a subgoal turns out impossible).
- **Tier L — local think-before-act (before every Actor step).** Within the current subgoal,
  the Reasoner does a focused think on just the next move and posts a `directive`/`correction`
  (ReAct). This is the per-step loop below.

The Reasoner does **not** post one insight and walk away. It **watches every Actor step and
intervenes the moment it sees an error or a wrong move** — like a pair-programming partner.

0. **Reasoner (global)** posts the `plan` node (Tier G) — runs once at task start.
1. **Reasoner (local)** reads the `plan` + shared graph (recent outcomes + facts) + its
   private graph → posts a **`directive`** node for the current subgoal: the hypothesis *and*
   a tool plan — *what kind of tool and which tools the Actor should use next, in order*
   (e.g. `{insight: "initials = first letter of each word, upper", plan: ["read_file buggy.py", "write_file", "run_tests"]}`).
2. **Actor** reads the latest `directive` + task → emits the next tool call + code →
   harness executes → verifier runs.
   - **If unsure** at any point, the Actor does NOT guess — it posts a **`query`** node
     ("which file is the bug in?", "uppercase the initials or not?"). The Reasoner answers
     with the next `directive`. This is v12 *clarifying constantly* instead of acting on a
     wrong assumption.
3. **Actor** posts a `result` node (pass/fail + error), edge `directive --led_to--> result`.
   **Every Actor action and result is appended to the shared graph LIVE — the instant it
   happens, not batched at the end.** So whenever the Reasoner reads, it sees the *current*
   state. The live graph IS the shared context — this is what guarantees the Reasoner never
   reasons on stale information.
4. **Reasoner monitors** the `result` *every step*. Four outcomes:
   - **all tests pass** → STOP (success).
   - **subgoal done, plan has more** → advance to the next subgoal in the Tier-G `plan`,
     go to step 1 with fresh local think. (Plan stays; only the local directive changes.)
   - **error / wrong step** (test fail, wrong tool used, no progress) → Reasoner posts a
     **`correction`** node immediately (edge `result --refuted--> directive`) — a new
     hypothesis + corrected tool plan. Loop. *This is the harness correcting v12.*
   - **stuck** (correction ladder exhausted, see Guards) → re-think Tier G (replan once),
     then **escalate to the parent**.

### Escalation ladder (loop engineering — try cheap, climb on failure)
1. **Tier 1 — Actor retry-with-hint:** failing assertion fed straight back (already in `loop.py`).
2. **Tier 2 — Reasoner correction:** the monitor posts a corrected directive + tool plan
   (the dual-brain's main new power; no bigger model).
3. **Tier 3 — Actor best-of-N:** N local v12 samples at rising temperature (already in `loop.py`).
4. **Tier 4 — call the parent:** hand the stuck file to `BIG_MODEL_ENDPOINT` if configured
   (already plumbed in `llm.chat_big`). Optional, off by default, fully local stays $0.

Each tier only fires when the one below it fails — the verifier decides "failed."

### Guards (designed against "both models looping")
- **Verifier early-exit** — tests pass ends everything (already in `loop.py`).
- **Round budget** — hard cap `MAX_ROUNDS` (e.g. 6) of reasoner↔actor exchange.
- **No-progress detector** — if 2 consecutive directives are ~identical (cosine > 0.95) or 2
  results identical → climb the escalation ladder, then fall back to single-brain `run()`.
- **Correction cap** — at most `MAX_CORRECTIONS` (e.g. 3) Reasoner corrections per directive
  before escalating to Tier 3/4. Stops the couple from bickering forever.
- **Graceful degrade** — any reasoner failure → Actor runs alone (today's proven path).

---

## 3. The shared graph module (NEW) — `agent/graph.py`

A lightweight, local, dependency-free graph. **Borrows the best ideas from other repos:**

- **From gbrain:** (1) *self-wiring typed edges* — extract entity/symbol refs on every write
  with **zero LLM calls** (regex/AST over node text); (2) *hybrid retrieval* — vector + BM25
  + graph-adjacency fused with **reciprocal-rank fusion** (gbrain credits the graph signal
  with +31% P@5 over vector-only); (3) *`think` vs `search`* — `search` returns raw nodes,
  `think` returns a synthesized answer + **gap analysis** ("what the graph doesn't know yet").
- **From GitNexus:** an **exact code-structure** read-source (callers/callees/impact) — used
  by the Reasoner to ground hypotheses in real code, not guesses. Already wired in
  `retriever.py`; the graph module *links to* it rather than re-implementing it.

### Data model
```
Node:
  id:        str (uuid-ish, deterministic from content hash — no Date/random)
  kind:      "plan" | "directive" | "correction" | "query" | "action" | "result" | "fact" | "symbol"
  author:    "reasoner" | "actor" | "system"
  text:      str
  payload:   dict (e.g. {tool, args, passed, error})
  vec:       float32[]  (lazy; via local embed server :8081)
Edge:
  src, dst:  node ids
  type:      "led_to" | "refuted" | "supports" | "about" | "calls"
  weight:    float
```

### API (mirrors EpisodicMemory so the loop already knows the shape)
```python
g = Graph(store_dir, embedder=default_embedder())
nid = g.add_node(kind, author, text, payload=None)      # auto-extracts edges
g.link(src, dst, type, weight=1.0)
hits = g.search(query, k, kinds=None)                   # hybrid vector+BM25+adjacency RRF
ans  = g.think(query)                                    # synthesized + gap analysis
g.neighbors(nid, type=None)                              # graph traversal
g.recent(kind, n)                                        # blackboard read
```

### Persistence
**Live append:** `add_node` writes to `nodes.jsonl` and updates the in-memory index
*synchronously on every call* — there is no batching. Both brains share one `Graph` object
for `shared/`, so an Actor write is visible to the next Reasoner read in the same process.
(Cross-process safety isn't needed — duo runs one process driving both model servers.)

`runtime/graphs/<name>/` → `nodes.jsonl` + `edges.jsonl` + `index.faiss` (reuse the FAISS
+ BM25 + IVF-PQ-at-scale machinery already in `episodic.py`; refactor the shared bits into a
small `vecindex.py` so both `episodic.py` and `graph.py` use one implementation).

### Three graph instances
- `graphs/reasoner/` — Reasoner's private graph.
- `graphs/actor/`    — Actor's private graph (≈ today's `episodic.py` store).
- `graphs/shared/`   — the blackboard.

---

## 4. New / changed code

| File | Change |
|---|---|
| `agent/vecindex.py` | NEW. Extract FAISS+BM25+RRF+IVF-PQ from `episodic.py` into a reusable index used by both `episodic.py` and `graph.py`. |
| `agent/graph.py` | NEW. The lightweight typed graph (§3): nodes/edges, self-wiring, hybrid search, `think`. |
| `agent/llm.py` | Add a 2nd chat target: `chat_reasoner()` → base model server (its own port, e.g. :8082 chat-mode; :8081 is embeddings). `chat()` stays the Actor (v12 :8080). |
| `agent/duo.py` | NEW. The coupled orchestrator: the reasoner↔actor blackboard loop + guards (§2). Exposes `solve(task)` — the "one agent" façade. |
| `agent/loop.py` | Unchanged proven `run()`; `duo.solve()` calls `run()`-style execution for the Actor's turn and reuses verifier/escalation. Single-brain remains the fallback. |
| `eval/run_duo.py` | NEW. Benchmark the dual-brain vs single-brain on heldout + heldout_multi. |

### Servers at runtime
- `:8080` v12 (Actor, chat, GPU)
- `:8081` base (embeddings, CPU)  ← already up
- `:8082` base (Reasoner, chat, CPU or GPU)  ← NEW (same on-disk base GGUF, chat mode)

> Note: 3 model instances of a 3B. Base is ~1.8GB each; Reasoner can run CPU (`-ngl 0`) if
> GPU VRAM is tight, like the embed server. All local, no download.

---

## 5. Message schema on the blackboard (the contract)

Reasoner → shared graph (a `directive` carries the hypothesis **and the tool plan** — the
Reasoner tells the Actor *what kind of tool and which tools* to use, in order):
```json
{"kind":"directive","author":"reasoner",
 "text":"to_celsius drops the +1; initials must join first letters uppercased",
 "payload":{"targets":["to_celsius","initials"],"confidence":0.8,
            "plan":["read_file:buggy.py","write_file:buggy.py","run_tests"]}}
```
Actor → shared graph:
```json
{"kind":"result","author":"actor",
 "text":"wrote buggy.py; tests still fail on initials",
 "payload":{"passed":false,"error":"AssertionError: 'A' != 'AL'","tool_used":"write_file"}}
```
Actor → shared graph (clarification; fires the moment v12 is unsure, instead of guessing):
```json
{"kind":"query","author":"actor",
 "text":"two functions look buggy — fix to_celsius first or initials first?",
 "payload":{"blocking":true}}
```
Reasoner → shared graph (monitoring + answering queries; fires on error/wrong step OR query):
```json
{"kind":"correction","author":"reasoner",
 "text":"you uppercased only the first letter; map upper() over EACH initial",
 "payload":{"refutes":"<directive_id>","plan":["write_file:buggy.py","run_tests"]}}
```
The Actor's prompt is built from: task + latest `directive`/`correction` (insight + tool
plan) + failing test. The Reasoner's prompt is built from: task + current file + latest
`result` node + `think()` recall from its own graph. The Reasoner runs **after every Actor
result**, so it is monitoring continuously, not just at the start.

---

## 6. Build phases (each independently testable)

1. **Phase A — refactor:** extract `vecindex.py` from `episodic.py`; prove `episodic` tests
   still pass (no behaviour change). *Lowest risk, do first.*
2. **Phase B — graph module:** build `agent/graph.py` on `vecindex`; unit-test add/link/
   search/think/neighbors on a toy set.
3. **Phase C — reasoner server + client:** start base on :8082 chat-mode; add
   `llm.chat_reasoner()`; smoke-test it reasons long (no brevity training).
4. **Phase D — duo orchestrator:** `agent/duo.py` blackboard loop + guards; test on the
   two-bug `initials` task (the canonical hard case) with escalation OFF — prove the
   *coupling itself* cracks it, no bigger model.
5. **Phase E — benchmark:** `eval/run_duo.py` — dual vs single on heldout + heldout_multi,
   honest (brain/answer-key off). Expect: equal on easy, **higher on hard multi-step**.

**Success criterion:** the two-bug task that single-brain v12 fails → **passes** via the
dual-brain coupling, with no external/bigger model. That is the proof the architecture earns
its complexity.

> **RESULT (2026-06-23): MET.** `eval/run_duo.py` solved the two-bug `initials` task in
> 2 rounds / 238s, `via=coupling`, no bigger model. Winning design (not the original rigid
> directives — the base model echoes those): the Reasoner THINKS in free-form CoT and the
> Actor (v12) TRANSCRIBES the reasoning into one clean file; retry-with-hint feeds the failing
> assertion back between rounds. Requires the Reasoner on GPU (`-ngl 99`, fits alongside v12
> in 8GB Arc) — on CPU the loop is too slow/flaky and degrades to single-brain.

---

## 7. Risks & honest notes

- **More moving parts = more to break** (the user's exact worry). Mitigation: the graph-as-
  bus decoupling + verifier anchor + graceful degrade to single-brain. If duo underperforms,
  `run()` is always there.
- **Latency:** 2–3 model round-trips per step. Local = time, not money. Acceptable.
- **The hash/causal-LM embedder is weak** (semantic recall limited). The shared graph's
  vector signal inherits this until a real embed model is available (HF blocked → fetch via
  Kaggle if needed). BM25 + typed edges carry retrieval meanwhile.
- **Eval honesty:** the shared/private graphs must NOT contain the exact eval answers during
  held-out runs (same memorization-via-retrieval trap as `episodic`). Seed only analogous
  facts; keep `EPISODIC_OFF`-style guards.
- **Naming honesty.** "One agent" is true (single `solve()` façade). "A **6B model**" is
  **not** — these are two separate 3B forward passes talking through a narrow text channel
  (the graph), not 6B params in one attention pass. Say *"~6B params total, 2×VibeThinker-3B"*
  or *"a 2×3B dual-brain agent."* Claiming a monolithic 6B undercuts the project's own thesis
  ("the harness is the intelligence, not the weights"). Product framing as one agent is fine;
  benchmark claims must disclose the two instances.

---

## 8. Extensibility — skills as graph participants

A **skill** is anything that reads the shared graph and writes back to it. You add one
*without touching either brain* — the blackboard is the integration point ("thin harness,
fat skills"). Three kinds:

- **Tools the Actor can call** — register in `tools.py` + one line in the Actor's tool list.
  The Reasoner can then name it in a `directive` plan. Examples: `lint`, `typecheck`,
  `grep`/AST search, `run-subset-of-tests`, `git blame`.
- **Producers** — run after an Actor step, post `result`/`fact` nodes (e.g. a linter posts
  its errors). Both brains reason over the new signal for free.
- **Knowledge sources** — post `fact` nodes (docs lookup, API reference). Read like any fact.

**Contract:** a skill may *propose* (write nodes) but never *decide* — the verifier (tests)
stays the only source of truth, so a bad skill is at worst ignorable noise on the graph.

**Discipline (same as what got us to 100%):** add skills **one at a time, each behind its
own eval.** The weak 3B mis-calls tools under load; don't bolt on five at once. Each new
tool = more surface area the Actor can get wrong, so prove it pays before adding the next.

---

## 9. Open issues → concrete solutions (resolve in build)

Each issue below has a chosen, effective fix. These are the build contract for `duo.py`.

### 9.1 Turn-order / deadlock (blocking query stalls the loop)
**Fix — single-threaded scheduler with a fixed priority, no waiting.** `duo.py` owns the only
loop; the brains never block on each other. Each tick the scheduler picks the next actor by
priority on the shared graph's tail:
```
open query?      → Reasoner answers it      (priority 1)
fresh result?    → Reasoner monitors        (priority 2)
have directive?  → Actor executes next step (priority 3)
none of the above→ Reasoner does global/local think
```
Because it's one process draining a queue, "blocking" just means *"this node must be consumed
before any Actor step"* — a sort key, not a thread wait. Deadlock is structurally impossible.

### 9.2 Reasoner names a non-existent tool
**Fix — validate every plan step against the live tool registry; coerce or drop.** `tools.py`
already has the registry; expose `TOOLS = {name: fn}`. When a `directive`/`correction` lands,
`duo.py` filters `plan` to known tools, fuzzy-maps near-misses (`"read"→"read_file"`), and if a
step can't resolve, drops it and appends a `fact` node *"unknown tool X ignored"* so the
Reasoner learns. The Actor only ever receives a validated plan. Zero LLM cost — pure dict lookup.

### 9.3 Query / chatter loop (no test ever runs)
**Fix — per-subgoal query budget + forced action.** Counter `queries[subgoal]`; cap = 2. On the
3rd query the scheduler injects a directive *"insufficient info — make your best attempt and run
tests"* and forces an Actor step. The verifier then produces real signal, which is always more
useful than more talk. Pairs with the existing `MAX_CORRECTIONS`/`MAX_ROUNDS` caps.

### 9.4 Node-ID collisions (content-hash overwrite)
**Fix — composite deterministic ID, still no Date/random.** `id = sha1(f"{author}|{kind}|{seq}|{text}")[:16]`
where `seq` is a per-graph monotonic counter persisted in `meta.json` and incremented on every
`add_node`. Identical text at different times → different `seq` → different id. Deterministic
(resume-safe), collision-free, no clock/RNG.

### 9.5 Context windowing (graph feed blows up — the v12 context-bloat trap)
**Fix — bounded, role-shaped context budget per brain.** Never feed the whole graph. Each
prompt gets: **task + current subgoal + last K=3 nodes on the tail + top-M=4 `search()` hits**
for the query, hard-capped at `CTX_CHARS` (e.g. 2500). Actor leans recent (what just failed);
Reasoner leans retrieved (relevant past insight). This is the same lean-context discipline that
fixed v12 — applied to the bus, not the prompt.

### 9.6 Eval honesty for duo
**Fix — `DUO_EVAL` guard that hard-empties the answer channel.** `run_duo.py` sets a flag that
(a) skips loading any persisted `graphs/*` store, (b) seeds only *analogous* facts (never the
held-out solutions), (c) asserts at startup that no node text contains an eval problem's answer
string. Same posture as `EPISODIC_OFF`; makes memorization-via-retrieval impossible, not just
discouraged.

### 9.7 Reasoner server unverified (`:8082` may not reason long)
**Fix — Phase-C gate before any loop wiring.** A smoke test (`eval/smoke_reasoner.py`) sends a
known hard prompt to `:8082` and asserts the reply is genuinely long-form CoT (e.g. > 800 chars,
contains step markers) — proving the *base* GGUF is loaded, not v12. If it fails, the build
stops at Phase C. The duo loop is never wired to an unverified reasoner.

**Build order impact:** 9.4/9.5 land in Phase B (graph module), 9.7 in Phase C, 9.1/9.2/9.3 in
Phase D (orchestrator), 9.6 in Phase E (benchmark). None change the architecture — they harden it.
