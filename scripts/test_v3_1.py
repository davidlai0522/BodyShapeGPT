import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Rebuild the Model Architecture
# ==========================================
class LLMToSMPLRegressor(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.llm = base_model
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 10)
        )
        # Must be present even at inference — it's part of the saved state dict
        self.register_buffer(
            "loss_weights",
            torch.ones(10, dtype=torch.float32)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Mean pooling — must match training forward()
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_hidden = mean_hidden.to(torch.float32)

        return self.regressor(mean_hidden)
    
# ==========================================
# 2. Setup and Load Weights
# ==========================================
# MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_NAME = "Qwen/Qwen2.5-0.5B" 
CHECKPOINT_DIR = "smpl_regressor_checkpoints_v3_1/best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_llm = AutoModel.from_pretrained(
    MODEL_NAME, 
    device_map=DEVICE,
    torch_dtype=torch.float16
)

# Re-apply the EXACT same LoRA config used in training
lora_config = LoraConfig(
    r=12,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # must match training
    lora_dropout=0.05,
    bias="none",
)
base_llm = get_peft_model(base_llm, lora_config)

# Initialize full model
hidden_size = base_llm.config.hidden_size
model = LLMToSMPLRegressor(base_llm, hidden_size)

# Load the saved state dictionary (saved as safetensors by trainer.save_model)
print(f"Loading checkpoint from {CHECKPOINT_DIR}...")
state_dict = load_file(os.path.join(CHECKPOINT_DIR, "model.safetensors"), device=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval() # Set to evaluation mode (disables dropout, etc.)

# ==========================================
# 3. Inference Function
# ==========================================
def predict_smpl_shape(description):
    inputs = tokenizer(
        description,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)
    
    with torch.no_grad(): # No need to track gradients for inference
        predicted_betas = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        )
        
    # Convert tensor back to a standard Python list
    return predicted_betas[0].cpu().numpy().tolist()

# ==========================================
# 4. Acceptability Evaluation
# ==========================================
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

_range_df_cache: Optional[pd.DataFrame] = None

def _load_range_df(ranges_csv: str) -> pd.DataFrame:
    global _range_df_cache
    if _range_df_cache is None:
        _range_df_cache = pd.read_csv(ranges_csv)
    return _range_df_cache


@dataclass
class BetaAcceptabilityResult:
    description: str
    matched_keywords: list
    per_beta: list          # list of dicts, one per beta index
    n_acceptable: int       # number of betas within range
    n_constrained: int      # betas that had at least one keyword constraint
    acceptance_rate: float  # n_acceptable / n_constrained (or n_acceptable / 10 if none)
    overall_acceptable: bool

    def __str__(self):
        lines = [
            f"Description : {self.description}",
            f"Keywords    : {self.matched_keywords if self.matched_keywords else '(none matched)'}",
            f"Acceptance  : {self.n_acceptable}/{self.n_constrained} constrained betas in range"
            f"  ({self.acceptance_rate:.0%})  →  {'✅ PASS' if self.overall_acceptable else '❌ FAIL'}",
            "",
            f"  {'Beta':<10} {'Predicted':>10} {'Range':>22}  {'Status'}",
            f"  {'-'*10} {'-'*10} {'-'*22}  {'-'*10}",
        ]
        for b in self.per_beta:
            if b["constrained"]:
                rng = f"[{b['range_lo']:+.3f}, {b['range_hi']:+.3f}]"
                status = "✅ in range" if b["in_range"] else "❌ out of range"
            else:
                rng = "(unconstrained)"
                status = "—"
            lines.append(f"  {b['beta']:<10} {b['predicted']:>+10.3f} {rng:>22}  {status}")
        return "\n".join(lines)


def evaluate_beta_acceptability(
    description: str,
    predicted_betas: list,
    ranges_csv: str = os.path.join(os.path.dirname(__file__), "keyword_beta_ranges.csv"),
    weights_csv: str = os.path.join(os.path.dirname(__file__), "keyword_beta_weights.csv"),
    percentile_lo: int = 10,
    percentile_hi: int = 90,
    pass_threshold: float = 0.75,
    min_abs_r: float = 0.15,
) -> BetaAcceptabilityResult:
    """
    Evaluate whether predicted SMPL betas are within the acceptable range for a
    given text description.

    Args:
        description:      Free-text body description (same format as training data).
        predicted_betas:  List of 10 floats (model output).
        ranges_csv:       Path to keyword_beta_ranges.csv produced by body_shape_analysis.ipynb.
        weights_csv:      Path to keyword_beta_weights.csv (used to filter weak correlations).
        percentile_lo/hi: Which percentile band to use as the acceptable range (default P10–P90).
        pass_threshold:   Fraction of constrained betas that must be in-range to call overall PASS.
        min_abs_r:        Minimum |r| for a keyword to constrain a given beta (default 0.15).
                          Keywords with weaker correlation are ignored for that beta so they
                          don't artificially narrow the range via intersection.

    Returns:
        BetaAcceptabilityResult with per-beta details and overall verdict.
    """
    range_df = _load_range_df(ranges_csv)
    corr_df  = pd.read_csv(weights_csv)   # keyword, beta, abs_r, ...
    lo_col = f"p{percentile_lo}"
    hi_col = f"p{percentile_hi}"

    all_keywords = range_df["keyword"].unique().tolist()
    matched = [kw for kw in all_keywords if kw.lower() in description.lower()]
    # Remove sub-phrase collisions: drop keyword A if another matched keyword B
    # contains A as a substring (e.g. "tall" is a sub-phrase of "tall neck").
    matched = [kw for kw in matched
               if not any(kw != other and kw.lower() in other.lower()
                          for other in matched)]

    per_beta = []
    n_acceptable = 0
    n_constrained = 0

    for b_idx in range(10):
        beta_name = f"beta_{b_idx}"
        pred = predicted_betas[b_idx]

        if matched:
            # Only keep keywords that have strong enough correlation with THIS beta
            strong_kws = [
                kw for kw in matched
                if corr_df[
                    (corr_df["keyword"] == kw) & (corr_df["beta"] == beta_name)
                ]["abs_r"].values[0] >= min_abs_r
            ]
            if strong_kws:
                sub = range_df[(range_df["keyword"].isin(strong_kws)) & (range_df["beta"] == beta_name)]
                lo = sub[lo_col].max()   # intersection over strong keywords only
                hi = sub[hi_col].min()
                constrained = True
                n_constrained += 1
            else:
                lo, hi = float("-inf"), float("inf")
                constrained = False
        else:
            lo, hi = float("-inf"), float("inf")
            constrained = False

        in_range = lo <= pred <= hi
        if constrained and in_range:
            n_acceptable += 1

        per_beta.append({
            "beta":        beta_name,
            "predicted":   pred,
            "range_lo":    lo if constrained else None,
            "range_hi":    hi if constrained else None,
            "constrained": constrained,
            "in_range":    in_range if constrained else None,
        })

    denom = n_constrained if n_constrained > 0 else 10
    acceptance_rate = n_acceptable / denom
    overall_acceptable = acceptance_rate >= pass_threshold

    return BetaAcceptabilityResult(
        description=description,
        matched_keywords=matched,
        per_beta=per_beta,
        n_acceptable=n_acceptable,
        n_constrained=n_constrained,
        acceptance_rate=acceptance_rate,
        overall_acceptable=overall_acceptable,
    )


# ==========================================
# 5. Test It
# ==========================================
if __name__ == "__main__":
    test_descriptions = [
        "Person with an average height, tall neck, long arms, and broad shoulders.",
        "A very tall, highly muscular individual with a heavy build and thick legs.",
        "A petite frame with narrow shoulders, short stature, and low body mass."
    ]
    
    print("\n--- Running Inference + Acceptability Evaluation ---")
    for desc in test_descriptions:
        print(f"\nInput: {desc}")
        predicted = predict_smpl_shape(desc)
        formatted_betas = [f"{b:.3f}" for b in predicted]
        print(f"Output Betas: {formatted_betas}")
        result = evaluate_beta_acceptability(desc, predicted)
        print(result)