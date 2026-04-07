"""
train.py — Fine-tune Qwen2.5-3B-Instruct (or any causal LM) on the
BodyShapeGPT dataset using QLoRA + SFTTrainer.

Usage:
    python train.py [--model MODEL_ID] [--output OUTPUT_DIR] [--epochs N]

Examples:
    python train.py
    python train.py --model Qwen/Qwen2.5-3B-Instruct --epochs 5
    python train.py --model Qwen/Qwen2.5-1.5B-Instruct --output weights_1b5
"""

import argparse
import json
import os
import math
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL     = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA      = "BodyShapeGPT_dataset.jsonl"
DEFAULT_OUTPUT    = "weights_new"
DEFAULT_EPOCHS    = 5
DEFAULT_LR        = 2e-4
DEFAULT_BSZ       = 4           # per-device batch size
DEFAULT_GRAD_ACC  = 4           # effective batch = BSZ * GRAD_ACC = 16
DEFAULT_MAXLEN    = 256         # max tokens per sample (desc + 10 floats easily fits)
DEFAULT_VAL_SPLIT = 0.1         # 10% validation
DEFAULT_LORA_R    = 16
DEFAULT_LORA_A    = 32
DEFAULT_LORA_DROP = 0.05

PROMPT_TEMPLATE = "### Description: {description}\n ### Shape parameters: {shape_params}"


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


def format_sample(sample: dict) -> dict:
    """Convert a raw JSONL record into a single text prompt+completion string."""
    text = PROMPT_TEMPLATE.format(
        description=sample["description"].strip(),
        shape_params=sample["shape_params"].strip(),
    )
    return {"text": text}


def build_dataset(data_path: str, val_split: float):
    raw = load_jsonl(data_path)
    records = [format_sample(s) for s in raw]

    n_val = max(1, math.floor(len(records) * val_split))
    n_train = len(records) - n_val

    # Deterministic shuffle via seed before split
    import random
    random.seed(42)
    random.shuffle(records)

    train_records = records[:n_train]
    val_records   = records[n_train:]

    train_ds = Dataset.from_list(train_records)
    val_ds   = Dataset.from_list(val_records)

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",   # required for SFTTrainer packing
    )
    # Qwen tokenizer may not have a pad token; use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False            # disable KV-cache during training
    model.config.pretraining_tp = 1

    print(f"[Model] Loaded  |  dtype={model.config.torch_dtype}  "
          f"|  device_map=auto")
    return tokenizer, model


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Target all linear projection layers (works for Qwen2/Llama/Phi)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, val_ds = build_dataset(args.data, args.val_split)

    # ── Model ────────────────────────────────────────────────────────────────
    tokenizer, model = load_base_model(args.model)
    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    # ── SFT config ───────────────────────────────────────────────────────────
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
        bf16=True,                  # RTX 5070 supports bf16
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,              # keep samples separate (short sequences)
        report_to="none",           # set to "wandb" if you have W&B set up
        seed=42,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    print("\n[Train] Starting training…")
    print(f"  Model:       {args.model}")
    print(f"  Output:      {args.output}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size} × {args.grad_acc} acc = "
          f"{args.batch_size * args.grad_acc} effective")
    print(f"  LoRA r/α:    {args.lora_r}/{args.lora_alpha}\n")

    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n[Train] Done! Adapter weights saved to '{args.output}/'")
    print(f"  → Run inference with:  python demo.py --weights {args.output} "
          f"--model {args.model} \"your description here\"")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a small LLM for SMPL shape parameter generation."
    )
    p.add_argument("--model",        default=DEFAULT_MODEL,
                   help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--data",         default=DEFAULT_DATA,
                   help=f"Path to JSONL dataset (default: {DEFAULT_DATA})")
    p.add_argument("--output",       default=DEFAULT_OUTPUT,
                   help=f"Output directory for LoRA weights (default: {DEFAULT_OUTPUT})")
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size",   type=int,   default=DEFAULT_BSZ,
                   dest="batch_size")
    p.add_argument("--grad-acc",     type=int,   default=DEFAULT_GRAD_ACC,
                   dest="grad_acc")
    p.add_argument("--max-seq-len",  type=int,   default=DEFAULT_MAXLEN,
                   dest="max_seq_len")
    p.add_argument("--val-split",    type=float, default=DEFAULT_VAL_SPLIT,
                   dest="val_split")
    p.add_argument("--lora-r",       type=int,   default=DEFAULT_LORA_R,
                   dest="lora_r")
    p.add_argument("--lora-alpha",   type=int,   default=DEFAULT_LORA_A,
                   dest="lora_alpha")
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROP,
                   dest="lora_dropout")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
