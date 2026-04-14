import ast
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
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
        # The base LLM (wrapped with LoRA)
        self.llm = base_model
        
        # The regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 10)  # 10 SMPL shape parameters
        )
        
        # CHANGE 2: Uniform loss weights — no bias toward early betas until
        # you've validated the model learns at all. Re-introduce weighting later
        # if domain knowledge justifies it.
        self.register_buffer(
            "loss_weights",
            torch.ones(10, dtype=torch.float32)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        # CHANGE 3: Mean pooling over non-padding tokens instead of last-token
        # pooling. Gives a richer, more stable representation for regression.
        mask_expanded = attention_mask.unsqueeze(-1).float()           # (B, T, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)        # (B, H)
        mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, H)

        # CHANGE 1: Cast to float32 before the regressor head for numerical
        # stability in mixed-precision training.
        mean_hidden = mean_hidden.to(torch.float32)

        predicted_betas = self.regressor(mean_hidden)

        loss = None
        if labels is not None:
            # CHANGE 1 (continued): Ensure labels are float32 to prevent
            # silent underflow when fp16=True is set in TrainingArguments.
            labels = labels.to(torch.float32)
            raw_squared_error = (predicted_betas - labels) ** 2
            weighted_squared_error = raw_squared_error * self.loss_weights
            loss = weighted_squared_error.mean()

        return {"loss": loss, "logits": predicted_betas}


# ==========================================
# 2. Setup and Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B"
DATA_FILE = "./BodyShapeGPT_dataset.jsonl"
OUTPUT_DIR = "./smpl_regressor_checkpoints_v2"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Base Model
print("Loading base model...")
base_llm = AutoModel.from_pretrained(
    MODEL_NAME,
    # device_map="auto",
    torch_dtype=torch.float16
)

# CHANGE 4: Expanded target_modules to include k_proj and o_proj.
# For a regression task mapping text → continuous floats, more expressive
# LoRA adaptation helps the model restructure its representations.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
base_llm = get_peft_model(base_llm, lora_config)
base_llm.print_trainable_parameters()

# Initialize full model
hidden_size = base_llm.config.hidden_size
model = LLMToSMPLRegressor(base_llm, hidden_size)

# Ensure the regression head is fully trainable and in float32
for param in model.regressor.parameters():
    param.requires_grad = True
    param.data = param.data.to(torch.float32)

# NOTE: No model.to() call here — device_map="auto" handles device placement.

# ==========================================
# 3. Data Processing
# ==========================================
def preprocess_function(examples):
    # FIX: No return_tensors="pt" — datasets .map() uses Arrow serialization
    # which cannot store PyTorch tensors. Return plain Python lists.
    model_inputs = tokenizer(
        examples["description"],
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    # FIX: Store labels as plain Python lists, not torch tensors.
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

# ==========================================
# 4. Custom Data Collator
# ==========================================
def custom_collate_fn(features):
    batch = {
        # FIX: Convert from plain lists to tensors here, not in preprocess_function.
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float32),
    }
    return batch

# ==========================================
# 5. Training
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    num_train_epochs=20,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,        # Keep best checkpoint, not just last
    ddp_find_unused_parameters=False,  # needed with LoRA + custom head
    metric_for_best_model="eval_loss",
    logging_steps=10,
    fp16=True,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=custom_collate_fn,
)

print("Starting training...")
trainer.train()

# Save the final model
torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_smpl_regressor.pth")
print("Training complete and model saved.")