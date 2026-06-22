# Step 9 — Fine-tuning VibeThinker for native tool-calling

**Why:** the eval baseline is **67% pass@1**. Failures come from the model not being
trained for tool-calling — it reaches correct actions through ~2000-token `<think>`
ramblings that burn the 8-step budget. An SFT pass teaches the *policy* (tight
reason → one action) directly. Train once in the cloud; run inference on-prem.

This laptop **cannot train** (no CUDA GPU; we hold only the quantized GGUF). Everything
here runs unchanged on any CUDA box — a free Kaggle T4/P100 notebook is the target.

## 0. The dataset (already built — local, free, no GPU)

```bash
python -m train.build_sft
```

Produces (committed to `data/`):
| file | what |
|---|---|
| `data/sft_curated.jsonl` | 7 hand-authored gold examples: ideal trajectory per problem + the weak-spot lessons (factorial base case, filename discipline, finish-on-PASS) |
| `data/sft_traces.jsonl`  | 11 examples harvested from **real winning** eval trajectories, with the verbose `<think>` replaced by tight reasoning |
| `data/sft_all.jsonl`     | the two combined — what the trainer reads (88 assistant target turns) |

Every example is multi-turn chat (`system`/`user`/`assistant`), ends on an assistant
turn, and trains **only on assistant tokens**.

## 1. Train (on a CUDA box / Kaggle)

```bash
pip install -r train/requirements.txt
python train/finetune_lora.py \
    --base <HF repo or path to the fp16 base matching your GGUF> \
    --data data/sft_all.jsonl \
    --out  out/vibethinker-tooluse-lora --epochs 8
```

> **Pick the right `--base`.** It must be the **fp16 safetensors** of the *same* model
> as `runtime/models/VibeThinker-3B.Q4_K_M.gguf` (you cannot train the GGUF itself).
> Confirm the exact WeiboAI HF repo/revision your GGUF was converted from before
> training, or the adapter won't compose with your inference weights.

QLoRA (4-bit) fits a 3B base in <16 GB → a single T4 works. On P100/A100 add `--no_4bit`.

## 2. Merge the adapter → fp16

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("<base>", torch_dtype="float16")
merged = PeftModel.from_pretrained(base, "out/vibethinker-tooluse-lora").merge_and_unload()
merged.save_pretrained("out/vibethinker-tooluse-fp16")
AutoTokenizer.from_pretrained("<base>").save_pretrained("out/vibethinker-tooluse-fp16")
```

## 3. Convert back to GGUF for llama.cpp (on-prem inference)

```bash
python llama.cpp/convert_hf_to_gguf.py out/vibethinker-tooluse-fp16 \
    --outfile VibeThinker-3B-tooluse.f16.gguf
./llama-quantize VibeThinker-3B-tooluse.f16.gguf \
    VibeThinker-3B-tooluse.Q4_K_M.gguf Q4_K_M
```

Drop `VibeThinker-3B-tooluse.Q4_K_M.gguf` into `runtime/models/`, point the server at
it, and **re-run the exact same eval** to measure the gain:

```bash
python -m eval.run_eval 3        # compare against the 67% baseline
```

## 4. Escalation path (the other half of Step 9)

The fine-tune raises the floor; escalation handles the rest. Trigger in `agent/loop.py`
when the verifier keeps failing / the loop stalls:
- repeated identical actions or N steps with no test-state change → stop flailing;
- escalate to a larger model (or human) with the trajectory so far as context;
- record cost-per-task either way.

This keeps the cheap 3B on the common case and pays for a big model only on the rare
hard one — the "same architecture at every tier" principle.
