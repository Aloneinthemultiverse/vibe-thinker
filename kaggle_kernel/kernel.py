# VibeThinker tool-calling LoRA fine-tune + GGUF export. Auto-generated.
# base: WeiboAI/VibeThinker-3B (the actual fp16 model our GGUF came from). T4 / QLoRA.
import json, os, random, subprocess, sys

os.makedirs("/kaggle/working", exist_ok=True)

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("pip -q install -U 'transformers>=4.44' 'peft>=0.11' 'datasets>=2.19' "
   "'accelerate>=0.30' sentencepiece")
# Kaggle ships torchao 0.10.0; this PEFT rejects it (<0.16) during LoRA injection even
# though we never use it. Remove it so PEFT skips the torchao dispatcher. Tolerant call.
subprocess.run("pip -q uninstall -y torchao", shell=True)

# Kaggle's prebuilt torch supports sm_70+ only, so a P100 (sm_60) raises
# cudaErrorNoKernelImageForDevice. The API push can't pick a GPU type, so make the kernel
# self-sufficient: probe the GPU arch in a CHILD process (don't import torch into THIS
# one yet), and if the stock build lacks it, reinstall an official wheel that includes
# sm_60. Done LAST so this torch wins over anything the lib installs pulled.
_probe = subprocess.run([sys.executable, "-c",
    "import torch,sys;c='sm_%d%d'%torch.cuda.get_device_capability(0);"
    "print('PROBE GPU',torch.cuda.get_device_name(0),c,torch.cuda.get_arch_list());"
    "sys.exit(0 if c in torch.cuda.get_arch_list() else 7)"])
if _probe.returncode == 7:
    print("Stock torch lacks this GPU's arch; installing official CUDA wheel (sm_60+)...", flush=True)
    sh("pip -q install --force-reinstall --no-deps torch --index-url "
       "https://download.pytorch.org/whl/cu121")
    # Reinstalling torch leaves torchvision/torchaudio built against the OLD torch ->
    # they crash on import ("torchvision::nms does not exist") and cascade into
    # transformers. We use neither for text LoRA, so remove them.
    subprocess.run("pip -q uninstall -y torchvision torchaudio", shell=True)

import torch
_cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0), _cap,
      "| arches", torch.cuda.get_arch_list(), flush=True)
if _cap not in torch.cuda.get_arch_list():
    raise SystemExit(f"{_cap} still unsupported after torch reinstall ({torch.cuda.get_arch_list()}).")
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, TrainerCallback)

# THE base must be the actual VibeThinker-3B fp16 weights (the model our GGUF was
# converted from) — NOT its upstream base Qwen2.5-Coder-3B. Training the upstream base
# would discard VibeThinker's reasoning post-training and the adapter would not compose
# with the inference GGUF.
BASE = "WeiboAI/VibeThinker-3B"
IGNORE = -100

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=True)
assert tok.is_fast, "need a fast tokenizer for offset_mapping"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# CRITICAL: use the model's NATIVE chat template — llama-server applies this exact
# template at inference (/v1/chat/completions). Do NOT override it, or train and infer
# formats diverge. Fall back to ChatML only if the tokenizer ships none.
if not tok.chat_template:
    tok.chat_template = (
        "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + "
        "'<|im_end|>' + '\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")
print("chat_template source:", "native" if tok.chat_template else "fallback-chatml", flush=True)

# Load the FULL 15k mixed set (attached Kaggle dataset): 49 verified in-domain bug-fix
# examples + 15k diverse Glaive tool-calling (524 tools). Shuffle so in-domain is spread.
rows = [json.loads(l) for l in
        open("/kaggle/input/vibethinker-sft-15k/sft_15k.jsonl", encoding="utf-8") if l.strip()]
random.Random(0).shuffle(rows)
print(f"training on FULL {len(rows)} examples", flush=True)

def tokenize(example, max_len=2048):
    # Robust assistant-only masking: render the full conversation once, then locate
    # each assistant turn's content by string search and map char spans -> token spans
    # via offset_mapping. No prefix-stability assumption (the earlier token-diff method
    # misaligned on Qwen's real template). Requires a fast tokenizer (Qwen has one).
    msgs = example["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels = [IGNORE] * len(ids)
    cur = 0
    for m in msgs:
        if m["role"] != "assistant":
            continue
        start = text.find(m["content"], cur)
        if start < 0:
            continue
        # Train through the turn terminator up to the next turn header (teaches stop).
        nxt = text.find("<|im_start|>", start + len(m["content"]))
        end = nxt if nxt != -1 else len(text)
        cur = end
        for j, (a, b) in enumerate(offs):
            if b <= start:
                continue
            if a >= end:
                break
            labels[j] = ids[j]
    assert len(ids) == len(labels)
    if sum(l != IGNORE for l in labels) == 0:
        raise ValueError("no assistant tokens labelled — chat-template/format mismatch")
    return {"input_ids": ids[:max_len], "labels": labels[:max_len],
            "attention_mask": [1] * len(ids[:max_len])}

ds = Dataset.from_list(rows).map(tokenize, remove_columns=["messages"])

def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    pid = tok.pad_token_id
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        p = L - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pid]*p)
        out["labels"].append(b["labels"] + [IGNORE]*p)
        out["attention_mask"].append(b["attention_mask"] + [0]*p)
    return {k: torch.tensor(v) for k, v in out.items()}

# Plain fp16 LoRA (NOT QLoRA). Kaggle may assign a Tesla P100 (CUDA sm_60), and
# bitsandbytes 4-bit kernels require sm_70+ ("named symbol not found" in ops.cu on
# P100). VibeThinker-3B fp16 (~6.5 GB) + LoRA + gradient checkpointing fits the 16 GB
# P100 fine, and this path runs on any GPU Kaggle hands us.
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                             device_map="auto", trust_remote_code=True)
# LoRA + gradient checkpointing on a non-quantized base needs inputs to require grad
# (prepare_model_for_kbit_training used to do this). Trainer enables the checkpointing.
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()

# Live training-loss logger: prints loss EVERY optimizer step and keeps the curve, so we
# can watch it fall in the Kaggle log and confirm the run actually learned.
class LossLogger(TrainerCallback):
    def __init__(self):
        self.losses = []
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            print(f"  [trainloss] step {state.global_step}/{state.max_steps} "
                  f"loss={logs['loss']:.4f}", flush=True)

_loss = LossLogger()
# 15k examples -> 1 epoch is plenty for SFT (more would overfit and exceed the ~12h
# session wall). batch=2/accum=4 (eff 16) keeps the P100 busy; log every step.
Trainer(model=model,
        args=TrainingArguments(output_dir="/kaggle/working/ckpt", num_train_epochs=1,
            per_device_train_batch_size=2, gradient_accumulation_steps=4, learning_rate=2e-4,
            lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="no",
            fp16=True, report_to="none", gradient_checkpointing=True),
        train_dataset=ds, data_collator=collate, callbacks=[_loss]).train()

if _loss.losses:
    print(f"[trainloss] curve: first={_loss.losses[0]:.4f} "
          f"min={min(_loss.losses):.4f} last={_loss.losses[-1]:.4f} "
          f"steps={len(_loss.losses)}", flush=True)

# Guaranteed small artifact: the LoRA adapter.
model.save_pretrained("/kaggle/working/adapter")
tok.save_pretrained("/kaggle/working/adapter")
print("adapter saved", flush=True)

# Merge to fp16 then convert -> GGUF (best effort; adapter already safe).
try:
    del model; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                                device_map="cpu", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, "/kaggle/working/adapter").merge_and_unload()
    merged.save_pretrained("/kaggle/working/merged", safe_serialization=True)
    tok.save_pretrained("/kaggle/working/merged")
    sh("git clone --depth 1 https://github.com/ggerganov/llama.cpp /kaggle/working/llama.cpp")
    sh("pip -q install -r /kaggle/working/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
    sh("python /kaggle/working/llama.cpp/convert_hf_to_gguf.py /kaggle/working/merged "
       "--outfile /kaggle/working/VibeThinker-3B-tooluse.f16.gguf --outtype f16")
    # Build only the quantize tool, then make Q4_K_M (matches the original GGUF).
    sh("cmake -S /kaggle/working/llama.cpp -B /kaggle/working/llama.cpp/build "
       "-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF >/dev/null 2>&1")
    sh("cmake --build /kaggle/working/llama.cpp/build --target llama-quantize -j2 >/dev/null 2>&1")
    sh("/kaggle/working/llama.cpp/build/bin/llama-quantize "
       "/kaggle/working/VibeThinker-3B-tooluse.f16.gguf "
       "/kaggle/working/VibeThinker-3B-tooluse.Q4_K_M.gguf Q4_K_M")
    os.remove("/kaggle/working/VibeThinker-3B-tooluse.f16.gguf")  # keep output small
    # Drop the big merged dir so it is not uploaded as output.
    sh("rm -rf /kaggle/working/merged /kaggle/working/llama.cpp /kaggle/working/ckpt")
    print("GGUF READY: VibeThinker-3B-tooluse.Q4_K_M.gguf", flush=True)
except Exception as e:
    print("GGUF conversion failed (adapter still saved):", repr(e), flush=True)
