"""
demo.py — Run inference with any fine-tuned LoRA model.

Usage (Gemma 4 E4B weights):
    python demo.py "Average height person with broad shoulders"

Usage (custom trained weights from train.py):
    python demo.py --model google/gemma-4-E4B-it --weights weights_new \\
                   "Average height person with broad shoulders"
"""

import argparse
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

DEFAULT_BASE_MODEL    = "google/gemma-4-E4B-it"
DEFAULT_WEIGHTS_DIR   = "weights"
SYSTEM_PROMPT         = "Convert the body description into 10 SMPL-X shape parameters."


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
        )

    processor = AutoProcessor.from_pretrained(
        model_id,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ft_model = PeftModel.from_pretrained(base_model, weights_dir)
    print("[Load] Complete\n")
    return processor, ft_model


def run_model(description: str, processor, ft_model, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_input = processor(text=prompt, return_tensors="pt").to(ft_model.device)
    ft_model.eval()
    with torch.no_grad():
        output_ids = ft_model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True)


def parse_betas(text: str) -> list[float]:
    """Extract the first 10 floats from the model response."""
    floats = re.findall(r"-?\d+\.\d+", text)
    betas = [float(x) for x in floats]
    return betas[:10]


def main():
    args = parse_args()

    processor, ft_model = load_model(args.model, args.weights, args.no_quantize)
    raw = run_model(args.description, processor, ft_model, args.max_new_tokens)
    betas = parse_betas(raw)

    print(f"Description : {args.description}")
    print(f"Raw output  : {raw!r}")
    print(f"Shape params: {betas}")
    if len(betas) < 10:
        print(f"[Warning] Only {len(betas)}/10 betas parsed.")


if __name__ == "__main__":
    main()
