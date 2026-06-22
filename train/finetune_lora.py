"""Step 9 — QLoRA fine-tune of VibeThinker-3B for native tool-calling discipline.

Trains ONLY on assistant turns (the policy), masking system/user/observation tokens
so the model learns to *produce* tight reason+action turns, not to parrot inputs.

This does NOT run on the dev laptop (no CUDA GPU; we hold only the quantized GGUF).
It is built to run unchanged on any CUDA box — a free Kaggle T4/P100 notebook is the
intended target. Inference stays on-prem (llama.cpp Vulkan); you train once in the
cloud, merge, convert to GGUF, and drop it into runtime/models/.

Pipeline:
    base safetensors (HF)  --LoRA SFT-->  adapter  --merge-->  fp16  --convert-->  GGUF

Usage (on a CUDA machine, after `pip install -r train/requirements.txt`):
    python train/finetune_lora.py \
        --base WeiboAI/VibeThinker-1.5B \
        --data data/sft_all.jsonl \
        --out  out/vibethinker-tooluse-lora

Then merge + convert (see train/README.md).
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

# VibeThinker is a Qwen2.5 derivative; -100 is the HF "ignore in loss" label.
IGNORE = -100


def load_examples(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tokenize_fn(tok, max_len=2048):
    """Assistant-only masking via render + string-search + offset mapping.

    Render the full conversation once, then locate each assistant turn's content in
    the rendered text and map its char span to token indices using offset_mapping.
    This makes no prefix-stability assumption about the chat template (an earlier
    token-diff method misaligned on Qwen's real template). Requires a fast tokenizer.
    """
    def fn(example):
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
    return fn


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [IGNORE] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="WeiboAI/VibeThinker-1.5B",
                    help="HF repo or local path to base safetensors (fp16).")
    ap.add_argument("--data", default="data/sft_all.jsonl")
    ap.add_argument("--out", default="out/vibethinker-tooluse-lora")
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--no_4bit", action="store_true", help="full-precision LoRA (P100/A100)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True, use_fast=True)
    assert tok.is_fast, "need a fast tokenizer for offset_mapping"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = load_examples(args.data)
    ds = Dataset.from_list(rows).map(build_tokenize_fn(tok, args.max_len),
                                     remove_columns=["messages"])
    print(f"dataset: {len(ds)} examples")

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=1, save_strategy="epoch", bf16=False, fp16=True,
        report_to="none", gradient_checkpointing=True)
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=lambda b: collate(b, tok.pad_token_id)).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nLoRA adapter saved -> {args.out}")
    print("Next: merge + convert to GGUF (see train/README.md).")


if __name__ == "__main__":
    main()
