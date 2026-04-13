"""
train.py — Fine-tune Qwen2.5-3B (or any causal LM) on BodyShapeGPT.

Default mode is supervised fine-tuning (SFT), which is the strongest baseline
for paired description -> shape-parameter data.

Optional mode: GRPO reward finetuning.

Usage:
    python train.py [--model MODEL_ID] [--output OUTPUT_DIR] [--epochs N]
"""

import argparse
import json
import math
import random
import re
from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from measurement import BodyMeasurements

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL     = "Qwen/Qwen2.5-3B"
DEFAULT_DATA      = "BodyShapeGPT_dataset.jsonl"
DEFAULT_OUTPUT    = "weights_qwen_base"
DEFAULT_EPOCHS    = 3
DEFAULT_LR        = 1e-4
DEFAULT_BSZ       = 4
DEFAULT_GRAD_ACC  = 4
DEFAULT_MAXLEN    = 256
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_LORA_R    = 64
DEFAULT_LORA_A    = 128
DEFAULT_LORA_DROP = 0.05
DEFAULT_NUM_GENS  = 4
DEFAULT_TRAINER   = "sft"
DEFAULT_WARMUP_STEPS = 100

# Timestamp for output directory naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

def parse_betas_from_text(text: str) -> list[float]:
    """Extract floats from a string (supports ints and scientific notation)."""
    floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    betas = [float(x) for x in floats]
    return betas[:10]


def canonical_betas_text(text: str) -> str:
    betas = parse_betas_from_text(text)
    betas = betas[:10] + [0.0] * (10 - len(betas))
    return "[" + ", ".join(f"{b:.6f}" for b in betas) + "]"


def split_records(records: list[dict], val_split: float) -> tuple[list[dict], list[dict]]:
    n_val = max(1, math.floor(len(records) * val_split))
    n_train = len(records) - n_val

    random.seed(42)
    random.shuffle(records)

    return records[:n_train], records[n_train:]


def format_sample_sft(sample: dict) -> dict:
    target = canonical_betas_text(sample["shape_params"])
    text = (
        f"### Description: {sample['description'].strip()}\n"
        f"### Shape parameters:\n"
        f"{target}"
    )
    return {"text": text}

def format_sample_grpo(sample: dict) -> dict:
    # GRPOTrainer feeds 'prompt' into the model for generations.
    prompt = (
        f"### Description: {sample['description'].strip()}\n"
        "### Shape parameters:\n"
        "Return exactly 10 comma-separated floats inside square brackets.\n"
        "Example: [0.123456, -0.234567, ..., 0.000000]\n"
    )

    gt_betas = parse_betas_from_text(sample['shape_params'])
    while len(gt_betas) < 10:
        gt_betas.append(0.0)

    return {
        "prompt": prompt,
        "gt_betas": gt_betas,
        "gt_text": canonical_betas_text(sample['shape_params']),
    }


def build_sft_dataset(data_path: str, val_split: float):
    raw = load_jsonl(data_path)
    records = [format_sample_sft(s) for s in raw]
    train_records, val_records = split_records(records, val_split)

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    print(f"[Data] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    return train_ds, val_ds


def build_grpo_dataset(data_path: str, val_split: float):
    raw = load_jsonl(data_path)
    records = [format_sample_grpo(s) for s in raw]
    train_records, val_records = split_records(records, val_split)

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

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

    model_dtype = getattr(model.config, "dtype", None)
    if model_dtype is None:
        model_dtype = getattr(model.config, "torch_dtype", None)
    print(f"[Model] Loaded  |  dtype={model_dtype}  |  device_map=auto")
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
# Reward Functions for GRPO
# ─────────────────────────────────────────────────────────────────────────────

def _get_text(comp):
    if isinstance(comp, str):
        return comp
    if isinstance(comp, dict):
        return str(comp.get("content", comp.get("text", comp)))
    if isinstance(comp, list) and len(comp) > 0:
        tail = comp[-1]
        if isinstance(tail, dict):
            return str(tail.get("content", tail.get("text", tail)))
        return str(tail)
    return str(comp)

def format_reward_func(completions, **kwargs):
    """Reward for generating exactly 10 parseable floats."""
    rewards = []
    for comp in completions:
        comp_text = _get_text(comp)
        betas = parse_betas_from_text(comp_text)
        if len(betas) == 10:
            rewards.append(1.0)
        else:
            rewards.append(-5.0)
    return rewards

def shape_l1_reward_func(completions, gt_betas, **kwargs):
    """L1 continuous shape error reward."""
    rewards = []
    for comp, gt in zip(completions, gt_betas):
        comp_text = _get_text(comp)
        pred = parse_betas_from_text(comp_text)
        pred = pred[:10] + [0.0] * (10 - len(pred))

        pred_t = torch.tensor(pred, dtype=torch.float32)
        gt_t = torch.tensor(gt, dtype=torch.float32)

        l1_diff = torch.nn.functional.l1_loss(pred_t, gt_t).item()
        rewards.append(-l1_diff)
    return rewards

# Global instance of BodyMeasurements to avoid reloading SMPL every batch
MEASUREMENTS_API = None

def get_measurements_api():
    global MEASUREMENTS_API
    if MEASUREMENTS_API is None:
        MEASUREMENTS_API = BodyMeasurements()
    return MEASUREMENTS_API

def measurement_reward_func(completions, gt_betas, **kwargs):
    """Measurement reward based on coarse body-trait consistency."""
    api = get_measurements_api()

    pred_batch = []
    gt_batch = []

    for comp, gt in zip(completions, gt_betas):
        comp_text = _get_text(comp)
        pred = parse_betas_from_text(comp_text)
        pred = pred[:10] + [0.0] * (10 - len(pred))
        pred_batch.append(pred)
        gt_batch.append(gt)

    beta_hat = torch.tensor(pred_batch, dtype=torch.float32, device=api.device)
    beta_gt = torch.tensor(gt_batch, dtype=torch.float32, device=api.device)

    with torch.no_grad():
        reward_t = api.compute_reward(beta_hat, beta_gt)

    return reward_t.cpu().tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_sft(args):
    processor, tokenizer, model = load_base_model(args.model)
    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    train_ds, val_ds = build_sft_dataset(args.data, args.val_split)

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
        warmup_steps=args.warmup_steps,
        fp16=False,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=processor,
    )

    print("\n[Train] Starting SFT training...")
    print(f"  Model:       {args.model}")
    print(f"  Output:      {args.output}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size} x {args.grad_acc} = {args.batch_size * args.grad_acc} effective")

    trainer.train()

    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\n[Train] Done! Adapter weights saved to '{args.output}/'")


def train_grpo(args):
    processor, tokenizer, model = load_base_model(args.model)
    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    train_ds, val_ds = build_grpo_dataset(args.data, args.val_split)

    eval_batch_size = args.batch_size
    if eval_batch_size % args.num_generations != 0:
        eval_batch_size = (
            (eval_batch_size + args.num_generations - 1) // args.num_generations
        ) * args.num_generations
        print(
            "[Config] Adjusted eval batch size to satisfy GRPO divisibility: "
            f"{eval_batch_size} (num_generations={args.num_generations})"
        )

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        max_prompt_length=args.max_seq_len,
        max_completion_length=128,
        num_generations=args.num_generations,
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            shape_l1_reward_func,
            measurement_reward_func,
        ],
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=processor,
    )

    print("\n[Train] Starting GRPO training...")
    print(f"  Model:       {args.model}")
    print(f"  Output:      {args.output}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Num Gens:    {args.num_generations}")
    print(f"  Batch size:  {args.batch_size} x {args.grad_acc} = {args.batch_size * args.grad_acc} effective")

    trainer.train()

    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\n[Train] Done! Adapter weights saved to '{args.output}/'")


def train(args):
    if args.trainer == "sft":
        train_sft(args)
        return

    train_grpo(args)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune an LLM via GRPO for SMPL shape parameter generation.")
    p.add_argument("--trainer",      choices=["sft", "grpo"], default=DEFAULT_TRAINER,
                   help="Training mode. Use sft for paper-closer supervised training, grpo for optional RL finetuning.")
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--data",         default=DEFAULT_DATA)
    p.add_argument("--output",       default=f"{DEFAULT_OUTPUT}_{DEFAULT_TRAINER}_{timestamp}")
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size",   type=int,   default=DEFAULT_BSZ, dest="batch_size")
    p.add_argument("--grad-acc",     type=int,   default=DEFAULT_GRAD_ACC, dest="grad_acc")
    p.add_argument("--max-seq-len",  type=int,   default=DEFAULT_MAXLEN, dest="max_seq_len")
    p.add_argument("--warmup-steps", type=int,   default=DEFAULT_WARMUP_STEPS, dest="warmup_steps")
    p.add_argument("--val-split",    type=float, default=DEFAULT_VAL_SPLIT, dest="val_split")
    p.add_argument("--lora-r",       type=int,   default=DEFAULT_LORA_R, dest="lora_r")
    p.add_argument("--lora-alpha",   type=int,   default=DEFAULT_LORA_A, dest="lora_alpha")
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROP, dest="lora_dropout")
    p.add_argument("--num-gens",     type=int,   default=DEFAULT_NUM_GENS, dest="num_generations",
                   help="Number of completions to generate per prompt for GRPO.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)