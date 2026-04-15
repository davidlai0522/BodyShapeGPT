import ast
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 1. Define the Custom Regression Model
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

        # Weights derived from correlation analysis:
        # β0-β3: strong keyword signal (r~0.4-0.5) → high weight
        # β5:    moderate signal (r~0.2)            → medium weight
        # β4,β8,β9: weak signal (r~0.1)            → low weight
        # β6,β7: no meaningful signal (r<0.06)     → zero (don't penalize)
        # Normalized to sum=10 for comparable loss scale across runs.
        self.register_buffer(
            "loss_weights",
            torch.tensor([
                2.27,  # β0 — strong (tall/short, r=0.49)
                2.27,  # β1 — strong (shoulders, r=0.43)
                2.27,  # β2 — strong (arms, r=0.43)
                2.27,  # β3 — strong (legs, r=0.41)
                0.45,  # β4 — weak
                0.91,  # β5 — moderate (legs secondary, r=0.21)
                0.00,  # β6 — no signal, excluded from loss
                0.00,  # β7 — no signal, excluded from loss
                0.45,  # β8 — weak
                0.11,  # β9 — weak
            ], dtype=torch.float32)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # Cast to float32 for numerical stability in mixed-precision training
        mean_hidden = mean_hidden.to(torch.float32)
        predicted_betas = self.regressor(mean_hidden)

        loss = None
        if labels is not None:
            labels = labels.to(torch.float32)
            raw_squared_error = (predicted_betas - labels) ** 2
            weighted_squared_error = raw_squared_error * self.loss_weights
            loss = weighted_squared_error.mean()

        return {"loss": loss, "logits": predicted_betas}


# ==========================================
# 2. Callback: Freeze LLM for first N epochs
#    Lets the regression head stabilize before
#    the LLM receives gradients, reducing noise
#    early in training.
# ==========================================
class UnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch=2):
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if not self._unfrozen and state.epoch >= self.unfreeze_epoch:
            print(f"\n[UnfreezeCallback] Unfreezing LLM at epoch {state.epoch:.0f}")
            # for param in model.llm.parameters():
            #     param.requires_grad = True
            self._unfrozen = True


# ==========================================
# 3. Setup and Configuration
# ==========================================
# MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_NAME = "Qwen/Qwen2.5-0.5B" 
DATA_FILE = "./BodyShapeGPT_dataset.jsonl"
OUTPUT_DIR = "./smpl_regressor_checkpoints_v3_1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_llm = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

# CHANGE: Reduced r=8 (from 16) to lower memorization capacity.
# lora_alpha kept at 2x r as convention.
lora_config = LoraConfig(
    # r=8,
    r=12,          # between 8 (underfit) and 16 (overfit)
    # lora_alpha=16,
    lora_alpha=24,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
base_llm = get_peft_model(base_llm, lora_config)
base_llm.print_trainable_parameters()

hidden_size = base_llm.config.hidden_size
model = LLMToSMPLRegressor(base_llm, hidden_size)

# Regressor head in float32 for stability
for param in model.regressor.parameters():
    param.requires_grad = True
    param.data = param.data.to(torch.float32)

# Freeze LLM initially — UnfreezeCallback will unfreeze at epoch 2
for param in model.llm.parameters():
    param.requires_grad = False


# ==========================================
# 4. Data Processing
# ==========================================
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["description"],
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    parsed_params = [ast.literal_eval(param_str) for param_str in examples["shape_params"]]
    model_inputs["labels"] = parsed_params
    return model_inputs

print("Loading and preprocessing dataset...")
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

tokenized_train = split_dataset["train"].map(
    preprocess_function, batched=True, remove_columns=["description", "shape_params"]
)
tokenized_eval = split_dataset["test"].map(
    preprocess_function, batched=True, remove_columns=["description", "shape_params"]
)

print(f"Train size: {len(tokenized_train)}, Eval size: {len(tokenized_eval)}")


# ==========================================
# 5. Custom Data Collator
# ==========================================
def custom_collate_fn(features):
    return {
        "input_ids":      torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels":         torch.tensor([f["labels"] for f in features], dtype=torch.float32),
    }


# ==========================================
# 6. Training
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # CHANGE: Larger batch reduces gradient noise → less overfitting
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    # Reduced from 3e-4 — the high LR caused grad_norm=inf (GradScaler overflow)
    # once the LLM was unfrozen at epoch 2, skipping most optimizer steps.
    # 5e-5 is standard for LoRA fine-tuning of 3B models.
    # learning_rate=5e-5,
    learning_rate=1e-4,

    # Keep high — early stopping will terminate before 20 if needed
    num_train_epochs=20,

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,             # shorter warmup since head is pre-warmed

    # CHANGE: weight_decay adds L2 regularization — penalizes large weights
    # and is the most direct overfitting countermeasure
    weight_decay=0.01,

    max_grad_norm=1.0,   # clip gradients, prevents the 100+ spikes

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Only keep the 3 best checkpoints to save disk space
    save_total_limit=3,

    logging_steps=10,
    bf16=True,   # bf16 has 8-bit exponent vs fp16's 5-bit → no overflow → no grad_norm=inf
    fp16=False,
    remove_unused_columns=False,
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/tb_logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=custom_collate_fn,
    callbacks=[
        # Stops training if eval_loss doesn't improve for 4 consecutive epochs
        EarlyStoppingCallback(early_stopping_patience=4),
        # Unfreezes LLM after epoch 2 once the regression head has stabilized
        # UnfreezeCallback(unfreeze_epoch=2),
    ]
)

print("Starting training...")
trainer.train()

# Use trainer.save_model instead of torch.save — this correctly saves
# the full model including LoRA adapter weights and tokenizer config
trainer.save_model(f"{OUTPUT_DIR}/best_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
print(f"Best model saved to {OUTPUT_DIR}/best_model")