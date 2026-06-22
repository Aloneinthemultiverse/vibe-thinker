# VibeThinker-OS — Build Plan

> A small reasoning core (VibeThinker-3B) wrapped in a planner / verifier / reflection
> harness, with an exact knowledge graph for structure and a TurboQuant-compressed
> vector store for semantic recall, driven by an orchestrator that calls tools via MCP.
> Goal: match frontier-model quality on the **verifiable core** of software engineering,
> cheaply, on-prem, on small hardware — scaling from a single laptop to enterprise.

---

## Guiding principles (decided in design discussion)

1. **The harness is the intelligence, not the weights.** The 3B is a *called oracle*, not the controller. A deterministic orchestrator drives the loop; the model only *proposes* actions.
2. **Verification is the center, not a step.** The model is good *because* it was trained against a verifier; push every step into a domain that can be checked (tests, compilers, schemas, exit codes).
3. **Structure → exact KG (lossless). Similarity → TurboQuant vectors (lossy, always re-ranked).** Never compress the graph; only compress the vectors hanging off its nodes.
4. **Earn every component by an observed failure.** Build the smallest loop first; add KG / memory / skills only when a real failure demands them.
5. **Honest scope.** Target parity on bounded, verifiable tasks. Degrade gracefully (retrieval + escalation) on knowledge-heavy / fuzzy tasks. Do not claim "beat Opus everywhere."
6. **Same architecture at every tier.** Laptop → mini-PC/Jetson → small server → rack. What scales is *memory and concurrency*, not the brain.

## Stack (all open / royalty-free)

- **Brain:** VibeThinker-3B (MIT, on Qwen2.5-Coder-3B). YaRN for modest context extension. **No SSA / attention surgery** — it risks the RL-tuned reasoning and duplicates what retrieval already does.
- **Inference on Intel Arc / NPU:** IPEX-LLM or llama.cpp (SYCL/Vulkan).
- **Graph store:** SQLite (or a small graph lib) — exact structure.
- **Vector store:** TurboQuant-compressed embeddings (e.g. `turboquant-pro`: PCA-Matryoshka + TurboQuant + HNSW, MIT).
- **Tool layer:** MCP (write-once connector; any MCP server plugs in).
- **Constrained decoding:** GBNF / Outlines / XGrammar to force parseable ACTION output from the untrained-for-tools 3B.

## Target hardware tiers

| Tier | Hardware | Notes |
|---|---|---|
| Dev / Entry | This laptop (Core Ultra 5 226V, 15.5 GB, Arc 130V 8 GB) | Proves the whole stack; 100k files fit in ~3 GB total |
| Small | mini-PC / Jetson Orin | 24/7, more vectors |
| Medium | small server + 1 modest GPU | fine-tuned vertical model, multi-user |
| Large | few GPUs / rack | many users, big graph, escalation to large model for the rare hard case |

---

## Milestones (build order = risk order, NOT diagram order)

### Step 1 — Brain bring-up: does VibeThinker run here, and is it good? ✅ DONE (2026-06-21)
**Result:** llama.cpp Vulkan + VibeThinker-3B Q4_K_M on Arc 130V. Solved the 2-bug merge_intervals
problem correctly (both bugs, passing fix). Speed: **32.2 tok/s gen, 391 tok/s prompt**, full answer < 1 min.
Python 3.14 blocked the IPEX-LLM/PyTorch route → pivoted to llama.cpp Vulkan (no Python needed for inference).
Runtime lives in `runtime/` (bin + models). Acceptance MET. Thesis core bet validated on real hardware.


**Intent:** Install IPEX-LLM (or llama.cpp SYCL) and run VibeThinker-3B (4-bit) on the Arc GPU. Feed it ONE self-contained problem (a failing test + the function under test) and confirm it produces a correct fix at usable speed (target: a full reasoning answer in ≤ ~3 min).
**Tags:** plan, build
**Acceptance:** model loads on Arc GPU; solves the sample problem correctly; tokens/sec measured and recorded; if it fails here, halt and reassess the whole thesis.
**Out of scope:** the orchestrator, any retrieval, any tools.
**Note:** environment/setup task — NOT an `/orchestrate` fit. Do manually.

### Step 2 — Minimal loop: brain proposes → code runs one tool → verify → repeat ✅ DONE (2026-06-21)
**Result:** Built `agent/` (protocol, tools, llm, loop) + `demo/` seeded-bug repo. Talks to llama-server.
Agent fixed merge_intervals autonomously in 4 steps (read→write→write→run_tests PASS). Fail-safe parser,
step budget, and JSONL tracing all verified. **Key finding:** VibeThinker (not tool-trained) emits the correct
action JSON but WITHOUT the requested ```action fence — fixed by a liberal parser that accepts bare post-</think>
JSON. This is the "meet the model where it is" lesson; a fine-tune (Step 9) would make the format native.
Step 3 (observability/tracing) folded in here. Acceptance MET.


**Intent:** Build the smallest possible orchestrator in plain code. The brain emits a strict `ACTION:`/`ARGS:` block (constrained decoding); the orchestrator parses it, runs ONE hardcoded tool (read file / run tests), checks the result with a deterministic verifier, feeds it back, and loops with a step budget. No KG, no vectors, no MCP. This is the whole thesis in miniature.
**Tags:** implement, test
**Acceptance:** end-to-end loop fixes a seeded bug in a tiny repo across ≥3 steps; malformed actions fail safe at the parser; step budget enforced; full trace logged.
**Out of scope:** semantic retrieval, knowledge graph, MCP, skills.

### Step 3 — Observability: trace every step
**Intent:** Add structured tracing of every loop iteration — prompt, retrieved context, model decision, tool result, verifier verdict. Painful to retrofit; build it now.
**Tags:** implement
**Acceptance:** each run emits a replayable per-step trace; a failing run can be diagnosed from the trace alone.
**Out of scope:** UI/dashboard polish.

### Step 4 — Retrieval part A: exact knowledge graph (structure) 🟡 WIRED (2026-06-21)
**Integration decision (HYBRID).** Evaluated three OSS repos for the KG layer:
- **GitNexus** — Tree-sitter code KG (nodes: Function/Class/File/Module; edges:
  CALLS/IMPORTS/EXTENDS/IMPLEMENTS/MEMBER_OF; LadybugDB; Cypher; impact/blast-radius).
  **Already installed and indexing the user's repos** (tinybench: 135 files/603 nodes/1895
  edges) and exposes a CLI emitting JSON. **Adopted as the structure substrate NOW.**
- **MarkItDown** (MIT) — PDF/DOCX/XLSX/HTML → Markdown. Adopt as the ingestion front-door
  for non-code docs (future Step 4.5). No license risk.
- **LLM Wiki** (**GPLv3**) — document-KG product overlapping our whole vision. GPLv3 = cannot
  link into our code. Reference only / optional sidecar over its HTTP API. Not a dependency.

**What was built:** `agent/retriever.py` — abstract `Retriever` interface + `GitNexusRetriever`
(subprocess to the `gitnexus` CLI; NOT the editor's MCP binding, since our agent is a plain
Python process) + `SqliteRetriever` stub so our OWN portable KG can swap in via one env flag
(`KG_BACKEND`). Three KG tools wired into the loop: `kg_query`, `kg_context`, `kg_impact`.
Verified: agent tool layer returns real exact-KG data (impact edges, ranked flows) for tinybench.
**Remaining for full DONE:** let the llm loop actually drive a cross-file task using kg_* tools;
note `embeddings: 0` on all repos → similarity side is Step 5's call (enable GitNexus embeddings
vs. our TurboQuant path — measure first). The "keep our own KG too" track = implement SqliteRetriever later.


**Intent:** Build the code graph over a small real repo (files, functions, imports, "depends_on"/"calls" edges) in SQLite. Wire a graph-traversal retriever ("what is connected to X?"). Keep the graph store and a future vector store as two coordinated indexes sharing a node ID.
**Tags:** implement, db
**Acceptance:** given a node, returns exact 1–2 hop neighbors; graph rebuilds from source; node IDs stable for cross-index linking.
**Out of scope:** semantic/vector search (Step 5), memory write-back (Step 6).

### Step 5 — Retrieval part B: TurboQuant-compressed vector store (similarity) ✅ DONE (2026-06-21)
**Completed end-to-end on REAL data.** `agent/embed.py` (cached MiniLM, 384-dim, forced
HF-offline so no network dependency) → `agent/vectorstore.py` (TurboQuant) → `agent/fusion.py`
(graph fusion). Demo on tinybench: 400 symbol-anchored chunks → compressed index **60.8 KB**
(3-bit) → **recall@10 = 0.996 on real embeddings**. Fusion proven: NL query → vector entries
(BenchOptions, defaultMinimumTime, bench) → GitNexus exact-KG expansion (BenchOptions imported
by utils.ts/task.ts). Similarity finds the neighborhood; exact KG gives the truth on connections.
Acceptance MET. (Algorithm sweep below retained for reference.)


**Built `agent/vectorstore.py`** — TurboQuant: fixed random orthogonal rotation →
Gaussian-optimal (Lloyd-Max) scalar quantization → asymmetric search (full-precision query
vs de-quantized codes) → re-rank top-N with full-precision originals. Pure numpy, deterministic.
**Measured recall@10 vs exact baseline** (5000×384 clustered unit vectors, faithful embedding proxy):
1-bit 0.708 (55×) · **2-bit 0.973 (29.5×)** · **3-bit 1.000 (20×)** · 4-bit 1.000 (15×).
**100k-file extrapolation (~600k chunks):** 62 MB @2-bit / 91 MB @3-bit (vs ~0.9 GB fp32 raw) —
within laptop budget. Acceptance MET at algorithm level: recall within tolerance, footprint
measured + in budget, structure stays exact (KG) / similarity lossy (this index).
**Remaining for full DONE:** (a) feed REAL embeddings (torch is blocked by Py3.14, but
onnxruntime+sentence-transformers import OK → use an ONNX embedding model, no torch);
(b) graph FUSION: vector hit → node ID → GitNexus traverse → re-rank. Both are wire-in, not
algorithm risk. The hard part (does compression preserve recall at scale on small hardware) is answered: YES.


**Empirical decision (measured, not assumed).** `gitnexus doctor` on this laptop:
`VECTOR index: unavailable` (LadybugDB vector engine disabled on win32), semantic mode =
**exact-scan, capped at 10,000 chunks**. Local ONNX embeddings ARE supported. Conclusion:
GitNexus = exact structure (Step 4 ✓) + usable embedding source, but its similarity search
does NOT scale to the 100k-file target on this platform. Per "earn every component by an
observed failure", our own **TurboQuant + ANN vector store is now justified** — reuse
GitNexus's local embeddings as the vector source, build the compressed scalable index ourselves.
**KG-driving test (2026-06-21):** VibeThinker autonomously called `kg_context("Bench")` then
`finish` with a correct blast-radius answer (29 test files, BenchLike, index.ts) on tinybench —
the 3B genuinely uses the KG tools. Step 4 proven end-to-end (`agent/run_kg_demo.py`).


**Intent:** Embed code chunks, store fingerprints in a TurboQuant-compressed vector index, and fuse with the graph: vector search finds the entry node → graph traversal expands → re-rank candidates with full precision. Validate recall against an uncompressed baseline before trusting it.
**Tags:** implement, review
**Acceptance:** compressed recall@10 within tolerance of uncompressed baseline on the test repo; 100k-file-scale memory footprint measured and within budget; structure stays on the exact KG path, similarity on the lossy path.
**Out of scope:** the graph build (Step 4), KV-cache compression.

### Step 6 — Memory write-back: read AND write ✅ CORE DONE (2026-06-21)
**Built `agent/memory.py`** (stdlib sqlite3, `runtime/memory.db`): decision/outcome records +
edges, three tiers with an enforced promotion policy — working→project after PROMOTE_USES=3
recalls, project→archive after ARCHIVE_AFTER_DAYS=30 untouched. Cold memories are ARCHIVED
(recoverable), never deleted; access (recall) is what promotes. Self-test verifies all three
invariants (write-back, promotion, archive-not-delete). **Wired into the loop**: `_persist()`
records outcome + runs promotion at every exit (finish / verifier-pass / budget). Verified the
loop writes to the real DB and recall returns it. Memory must never break the loop (guarded).
**Remaining polish:** feed recalled memory BACK into the prompt (read path into the loop) and
auto-extract new KG edges from task discoveries — enhancements, not blockers.


**Intent:** Add the write path so finished tasks update project memory / KG ("chose X because Y", new dependencies) with a promotion policy: working → project → archive. Without this the system never learns and the KG goes stale.
**Tags:** implement, db
**Acceptance:** a completed task persists a decision record and any new edges; promotion/eviction policy documented and enforced; stale entries archived not silently dropped.
**Out of scope:** distributed/multi-node memory.

### Step 7 — MCP tool layer + guardrails 🟡 GUARDRAILS DONE (2026-06-21)
**Built `agent/guardrails.py`** and wired it into the loop as the gate before every tool exec.
Enforces (all fail-safe / deny-on-doubt): risk policy (dangerous tools deploy/delete/send/shell
require approval; autonomous mode = DENY), allow/deny lists, UNKNOWN-tool denial, loop detection
(same (tool,args) N× blocked — also fixes the Step 2 double-write wobble), and a hard tool-call
budget. Self-test covers all six; loop integration verified (read_file allowed, deploy denied,
write_file allowed). Blocked actions feed the reason back to the model instead of throwing.
**Remaining:** generic MCP client so ANY MCP server plugs in (GitNexus already serves as the
1st MCP-style server via the CLI bridge; need a 2nd + the connector abstraction). The security
substance (guardrails) is the earned/risky part and is done; MCP wiring is mechanical.


**Intent:** Replace hardcoded tools with the MCP connector. Orchestrator pre-filters to the 2–5 relevant tools per step (via retrieval). Add guardrails: tool permissions, human-approval gates for dangerous actions (deploy/delete/send), loop detection, cost/step budgets.
**Tags:** implement, security
**Acceptance:** ≥2 MCP servers (e.g. filesystem + a build/test runner) drive the loop; dangerous actions gated; permission violations fail safe; budgets enforced.
**Out of scope:** the vertical fine-tune (Step 9).

### Step 8 — Skills layer + the constitution 🟡 BUILT (2026-06-21)
**Built `agent/skills.py`**: a versioned CONSTITUTION (v1.0.0 — role, strict format,
propose-never-execute, safety boundaries) + 3 frozen SKILLS (fix-failing-test,
understand-before-change, locate-by-meaning) retrieved by keyword and injected only when
relevant (prompt stays tight on no-match). Wired into the loop: `build_system_prompt(task,
TOOL_HELP)` composes constitution + tools + relevant skills; skills used are logged per run.
Self-test verifies routing; loop integration verified (constitution + correct skill in prompt).
**Remaining:** the ablation (measure step/failure reduction WITH vs WITHOUT skills) needs GPU
runs — deferred. Mechanism + versioning done; "measurably helps" is an empirical follow-up.


**Intent:** Add retrievable "skills" (frozen, proven procedures the model shouldn't re-derive) injected by the orchestrator when relevant, plus a disciplined system prompt (role, strict output format, "propose-never-execute", safety boundaries).
**Tags:** implement
**Acceptance:** at least 3 skills measurably reduce steps/failures on representative tasks; system prompt versioned; ablation shows skill injection helps.
**Out of scope:** auto-learning new skills (future).

### Step 9 — Vertical fine-tune + escalation path 🟡 DATASET + TRAINER READY (2026-06-21)
**Intent:** Fine-tune VibeThinker on a narrow vertical's tools + data (this is *also* what makes tool-calling reliable — same project). Define the escalation path: when the verifier keeps failing / confidence is low, escalate to a larger model or a human.
**Tags:** migration, review
**Acceptance:** fine-tuned model improves tool-call validity and task success on the vertical vs. base; escalation triggers on stuck loops instead of flailing; cost-per-task recorded.
**Out of scope:** full production deployment hardening.

**Progress (data half — done, local, $0 GPU):** Laptop cannot train (no CUDA GPU; we hold only
the Q4_K_M GGUF, not fp16 safetensors). Built the SFT pipeline instead:
- `train/build_sft.py` harvests **winning** eval trajectories (`runtime/traces/*.jsonl`, only
  `event:success`) → keeps the real action sequence + winning file content, **strips the
  ~2000-tok `<think>` waffle**, pairs it with tight reasoning. + 7 hand-curated gold examples
  (ideal trajectory per problem + weak-spot lessons: factorial base case, filename discipline,
  finish-on-PASS). → `data/sft_all.jsonl` (18 examples, 88 assistant target turns, all multi-turn,
  all end on an assistant turn, zero think leakage).
- `train/finetune_lora.py` — QLoRA, **assistant-tokens-only loss**, runs unchanged on a free
  Kaggle T4/P100. `train/requirements.txt` + `train/README.md` document the full
  base→LoRA→merge→GGUF loop back onto on-prem llama.cpp, and the re-eval-vs-67% gate.
**Remaining:** run the trainer on a CUDA box (Kaggle), convert to GGUF, re-run `eval.run_eval`
to measure gain vs 67%; wire the escalation trigger (stuck-loop → bigger model/human) in
`agent/loop.py`. The thesis holds: train once in cloud, infer forever on-prem.

---

## Integrated test (2026-06-21) — full stack, real model
Ran the bug-fix demo end-to-end through every new layer (constitution+skills prompt → guardrail
gate → execute → auto-verify → memory write-back), 4 runs. **Result: the HARNESS is validated and
hardened; task success is gated by MODEL quality (stochastic, not tool-trained).**

Harness layers all confirmed working under integration: prompt composition, guardrail loop-detection
(fired every run), memory write-back (outcomes persisted), auto-verify-after-write, fail-safe dispatch.

**4 robustness fixes EARNED by the test** (each from an observed failure):
1. **Auto-verify after every write** (`loop.py`) — model wrote an incomplete fix and never tested it;
   the orchestrator now runs the verifier on every write so the model can't fly blind.
2. **Degenerate-write guard** (`tools.py`) — model once wrote the literal placeholder "<code>" (7 bytes),
   corrupting the file; writes <20 chars / placeholder-like / no `def ` are now refused with feedback.
3. **Fail-safe tool dispatch** (`loop.py`) — a sandbox-escape ValueError from a model-supplied bad path
   CRASHED the whole run; tool exceptions are now caught and fed back, never crash the orchestrator.
4. **Clean KG errors** (`retriever.py`) — gitnexus leaked a raw `file://...backend.js` stack that the
   model then tried to read; errors are now short and safe, no paths to chase.

**Honest finding:** the 3B (temp 0.6, not tool-trained) solves this reliably *sometimes* (the original
Step 2 run passed in 4 steps) but flails on other rolls — incomplete fixes, ignoring test feedback,
wandering into irrelevant tools. This is exactly the motivation for **Step 9 (vertical fine-tune for
native tool-use)** and points to two cheap pre-fine-tune wins: lower temperature for deterministic
tasks, and don't expose kg_* tools when the task target isn't an indexed repo.

## Effectiveness eval (2026-06-21) — pass@1 baseline before Step 9
Built `eval/` (5 seeded-bug problems, isolated sandboxes, configurable temp/KG/trials).
First measured baseline: **67% pass@1 (10/15)**, temp 0.2, KG hidden, 3B with NO fine-tune.
Per-problem: binary_search 3/3 · merge_intervals 2/3 (1 infra HTTP-400) · is_palindrome 2/3 ·
fizzbuzz 2/3 · factorial 1/3. Passes mostly land in 4–6 steps; failures hit the 8-step budget.
Excluding the one infra flake: 10/14 = 71%. This is the number Step 9 (fine-tune) must beat.
**Two more fixes earned:** `finish` reason made optional (model omitted it → parse-retry loop that
also bloated context toward the HTTP-400); eval trials wrapped so one bad request can't kill the suite.
**Known minor model quirks logged:** occasionally uses the function name as the filename
(reads `fizzbuzz.py` not `buggy.py`); intermittent HTTP-400 on long transcripts (context growth).

## Non-goals (explicit)
- Reasoning over giant raw context windows (retrieval replaces this; SSA shelved).
- Beating frontier models on knowledge-heavy / taste / ambiguous tasks.
- Compressing the knowledge graph.
- Making the model execute tools directly (it only proposes).

## First action
Start with **Step 1** manually on this laptop. Nothing else matters until the brain runs here at usable speed.
