# CoupleVibe LLM

A local, $0 autonomous coding agent where **two tiny 3B models work as a couple** — a long-CoT
*Reasoner* and a tool-tuned *Actor* — coupled through a shared graph. Proof that **the harness
is the intelligence, not the weights.**

> **Why the name:** two brains, *coupled* like partners — one thinks, one acts — talking only
> through a shared graph. Built on the MIT VibeThinker-3B ("Vibe") lineage.
> *Technically it's a **2×3B dual-brain agent** (two LLMs + harness), branded CoupleVibe LLM.*

A raw VibeThinker-3B solves ~50–58% of held-out bugs. Inside the harness it hits **100% on
held-out single-bug tasks**; the **dual-brain coupling hits 100% on hard compound multi-bug
tasks** (vs 83% single-brain), every one solved *via coupling* with no bigger model — *no
fine-tuning, no memorization* (evals run honest). Runs locally on an Intel Arc GPU via
llama.cpp. No API, no cost. See [docs/dual-brain-design.md](docs/dual-brain-design.md).

## How it works

The model only ever **proposes one action**. A deterministic orchestrator runs it, **verifies
with tests after every write**, and early-exits the moment tests pass. The model never decides
success — the verifier does.

```
task ─► [model proposes ONE action] ─► orchestrator executes ─► run_tests
          ▲                                                        │
          └──── retry-with-hint / escalation ◄── fail ────────────┘
                                              pass ─► STOP
```

Escalation ladder when stuck: **retry-with-hint → best-of-N (local) → parent model** (optional
`BIG_MODEL_ENDPOINT`, off by default so it stays $0).

## Layout

| Path | What |
|---|---|
| `agent/loop.py` | Orchestrator: propose→execute→verify, retry-hint, escalation, planning. |
| `agent/llm.py` | The only place a model is called (v12 on `:8080`; optional parent). |
| `agent/tools.py` | Sandboxed tools: read/write_file, run_tests, KG lookups. |
| `agent/episodic.py` | Episodic "brain": FAISS + BM25 + GitNexus graph, RRF fusion, think-style recall. |
| `agent/retriever.py` | GitNexus code-graph wrapper (callers/callees/impact). |
| `eval/heldout.py` | 12 single-bug held-out problems (disjoint from training). |
| `eval/heldout_multi.py` | 6 compound 2-bug problems + analogous seeds. |
| `eval/ablation.py` | Raw-model vs full-harness ablation (the headline numbers). |
| `docs/dual-brain-design.md` | Design spec for the coupled 2×3B architecture. |

## Run

```bash
# 1. serve the model (llama.cpp, Vulkan on Arc)
llama-server -m VibeThinker-3B-tooluse.Q4_K_M.gguf --port 8080 -ngl 99

# 2. honest held-out eval (brain OFF — no answer-key recall)
python -m eval.run_eval

# 3. ablation: how much the harness adds over the raw model
python -m eval.ablation --single --heldout   # raw model, no harness
python -m eval.ablation         --heldout     # full harness
```

## Results (on-device, honest)

| Setup | Held-out |
|---|---|
| base 3B, agent-loop | 3% |
| base 3B, single-shot | 58% |
| v12, single-shot | 50% |
| **v12 + harness** | **100%** |
| v12 + harness, hard multi-bug | 83% |
| **dual-brain, hard multi-bug** | **100% (6/6, all via coupling)** |

The harness roughly **doubles** the raw model. That's the whole thesis. The dual-brain
(base Reasoner + v12 Actor coupled via a shared graph) then beats the single-brain harness
on hard compound bugs (83% → 100%), all local, no fine-tuning. See
[docs/dual-brain-design.md](docs/dual-brain-design.md).

## License

VibeThinker-3B is MIT (WeiboAI). This harness: see repository. Note: LLM Wiki is GPLv3 and is
**not** linked or imported here.
