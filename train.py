"""
train.py — Fine-tune Qwen2.5-3B-Instruct (or any causal LM) on the
BodyShapeGPT dataset using QLoRA + SFTTrainer.

Usage:
    python train.py [--model MODEL_ID] [--output OUTPUT_DIR] [--epochs N]
"""

import argparse
import json
import os
import math
import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL     = "Qwen/Qwen2.5-3B"
DEFAULT_DATA      = "BodyShapeGPT_dataset.jsonl"
DEFAULT_OUTPUT    = "weights_qwen_base_new"
DEFAULT_EPOCHS    = 10
DEFAULT_LR        = 2e-4
DEFAULT_BSZ       = 4           
DEFAULT_GRAD_ACC  = 4           
DEFAULT_MAXLEN    = 256         
DEFAULT_VAL_SPLIT = 0.1         
DEFAULT_LORA_R    = 64          
DEFAULT_LORA_A    = 128
DEFAULT_LORA_DROP = 0.05

# Removed the hardcoded <eos> and cleaned up the spacing
PROMPT_TEMPLATE = "### Description: {description}\n### Shape parameters: {shape_params}"

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[Data] Loaded {len(samples):,} samples from {path}")
    return samples

def format_sample(sample: dict, tokenizer) -> dict:
    # Moved the space to a newline to prevent tokenizer merging
    prompt = f"### Description: {sample['description'].strip()}\n### Shape parameters:\n"
    completion = f"{sample['shape_params'].strip()}{tokenizer.eos_token}"
    return {"prompt": prompt, "completion": completion}

def build_dataset(data_path: str, val_split: float, tokenizer):
    raw = load_jsonl(data_path)
    records = [format_sample(s, tokenizer) for s in raw]

    n_val = max(1, math.floor(len(records) * val_split))
    n_train = len(records) - n_val

    import random
    random.seed(42)
    random.shuffle(records)

    train_ds = Dataset.from_list(records[:n_train])
    val_ds   = Dataset.from_list(records[n_train:])

    print(f"[Data] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    return train_ds, val_ds

# ─────────────────────────────────────────────────────────────────────────────
# Model / tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(model_id: str):
    print(f"[Model] Loading base model: {model_id}  (4-bit QLoRA)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        padding_side="right",
    )
    
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    print(f"[Model] Loaded  |  dtype={model.config.torch_dtype}  |  device_map=auto")
    return processor, tokenizer, model

def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Model (Loaded first so we have the tokenizer for the dataset) ────────
    processor, tokenizer, model = load_base_model(args.model)
    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, val_ds = build_dataset(args.data, args.val_split, tokenizer)

   # ── Update SFT config ────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        completion_only_loss=True, 
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
        seed=42,
    )

    # ── Update Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=processor,
    )
    
    print("\n[Train] Starting training…")
    print(f"  Model:       {args.model}")
    print(f"  Output:      {args.output}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size} × {args.grad_acc} = {args.batch_size * args.grad_acc} effective")
    
    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\n[Train] Done! Adapter weights saved to '{args.output}/'")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a small LLM for SMPL shape parameter generation.")
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--data",         default=DEFAULT_DATA)
    p.add_argument("--output",       default=DEFAULT_OUTPUT)
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size",   type=int,   default=DEFAULT_BSZ, dest="batch_size")
    p.add_argument("--grad-acc",     type=int,   default=DEFAULT_GRAD_ACC, dest="grad_acc")
    p.add_argument("--max-seq-len",  type=int,   default=DEFAULT_MAXLEN, dest="max_seq_len")
    p.add_argument("--val-split",    type=float, default=DEFAULT_VAL_SPLIT, dest="val_split")
    p.add_argument("--lora-r",       type=int,   default=DEFAULT_LORA_R, dest="lora_r")
    p.add_argument("--lora-alpha",   type=int,   default=DEFAULT_LORA_A, dest="lora_alpha")
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROP, dest="lora_dropout")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)