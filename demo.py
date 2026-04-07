"""
demo.py — Run inference with any fine-tuned LoRA model.

Usage (original LLaMA-3-8B weights):
    python demo.py "Average height person with broad shoulders"

Usage (custom trained weights from train.py):
    python demo.py --model Qwen/Qwen2.5-3B-Instruct --weights weights_new \
                   "Average height person with broad shoulders"
"""

import argparse
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE_MODEL    = "meta-llama/Meta-Llama-3-8B"
DEFAULT_WEIGHTS_DIR   = "weights"
PROMPT_TEMPLATE       = "### Description: {description}\n ### Shape parameters: "


def parse_args():
    p = argparse.ArgumentParser(
        description="BodyShapeGPT inference — generate SMPL-X shape params from text."
    )
    p.add_argument("description", type=str,
                   help="Natural language description of the avatar body shape.")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL,
                   help=f"Base model HuggingFace ID (default: {DEFAULT_BASE_MODEL})")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS_DIR,
                   help=f"Path to LoRA adapter directory (default: {DEFAULT_WEIGHTS_DIR})")
    p.add_argument("--max-new-tokens", type=int, default=100, dest="max_new_tokens")
    p.add_argument("--no-quantize", action="store_true", dest="no_quantize",
                   help="Disable 4-bit quantization (requires more VRAM but faster on big GPUs)")
    return p.parse_args()


def load_model(model_id: str, weights_dir: str, no_quantize: bool):
    print(f"[Load] Base model : {model_id}")
    print(f"[Load] LoRA weights: {weights_dir}")

    if no_quantize:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ft_model = PeftModel.from_pretrained(base_model, weights_dir)
    print("[Load] Complete\n")
    return tokenizer, ft_model


def run_model(description: str, tokenizer, ft_model, max_new_tokens: int) -> str:
    prompt = PROMPT_TEMPLATE.format(description=description)
    model_input = tokenizer(prompt, return_tensors="pt").to(ft_model.device)
    ft_model.eval()
    with torch.no_grad():
        output_ids = ft_model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def parse_betas(text: str) -> list[float]:
    """Extract the first 10 floats from the model response."""
    floats = re.findall(r"-?\d+\.\d+", text)
    betas = [float(x) for x in floats]
    return betas[:10]


def main():
    args = parse_args()

    tokenizer, ft_model = load_model(args.model, args.weights, args.no_quantize)
    raw = run_model(args.description, tokenizer, ft_model, args.max_new_tokens)
    betas = parse_betas(raw)

    print(f"Description : {args.description}")
    print(f"Shape params: {betas}")
    if len(betas) < 10:
        print(f"[Warning] Only {len(betas)}/10 betas parsed — raw output: {raw!r}")


if __name__ == "__main__":
    main()
