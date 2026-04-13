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
            nn.Linear(512, 10) # 10 SMPL shape parameters
        )
        
        # Define weights for the Weighted MSE Loss
        # Prioritizing the first few parameters (overall size/proportions)
        self.register_buffer(
            "loss_weights", 
            torch.tensor([5.0, 5.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        # Grab the hidden state of the last token
        last_token_states = hidden_states[torch.arange(batch_size), sequence_lengths]
        
        # Cast to match the regressor's float32 dtype (prevents mixed-precision crashes)
        last_token_states = last_token_states.to(self.regressor[0].weight.dtype)
        
        predicted_betas = self.regressor(last_token_states)
        
        loss = None
        if labels is not None:
            # Weighted Mean Squared Error
            raw_squared_error = (predicted_betas - labels) ** 2
            weighted_squared_error = raw_squared_error * self.loss_weights
            loss = weighted_squared_error.mean()
            
        return {"loss": loss, "logits": predicted_betas}

# ==========================================
# 2. Setup and Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B" 
DATA_FILE = "/home/schaeffler/david_ws/test_ws/BodyShapeGPT/BodyShapeGPT_dataset.jsonl"
OUTPUT_DIR = "./smpl_regressor_checkpoints"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Base Model
print("Loading base model...")
base_llm = AutoModel.from_pretrained(
    MODEL_NAME, 
    device_map="auto",
    torch_dtype=torch.float16
)

# Apply LoRA to the base LLM
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
)
base_llm = get_peft_model(base_llm, lora_config)

# Initialize full model
hidden_size = base_llm.config.hidden_size
model = LLMToSMPLRegressor(base_llm, hidden_size)

# Ensure the regression head is fully trainable and in float32
for param in model.regressor.parameters():
    param.requires_grad = True
    param.data = param.data.to(torch.float32)

model.to(base_llm.device)

# ==========================================
# 3. Data Processing
# ==========================================
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["description"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    parsed_params = [ast.literal_eval(param_str) for param_str in examples["shape_params"]]
    model_inputs["labels"] = torch.tensor(parsed_params, dtype=torch.float32)
    
    return model_inputs

print("Loading and preprocessing dataset...")
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["description", "shape_params"])
tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=["description", "shape_params"])

# ==========================================
# 4. Custom Data Collator
# ==========================================
def custom_collate_fn(features):
    batch = {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "labels": torch.stack([torch.tensor(f["labels"]) for f in features])
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
    num_train_epochs=20,                 # INCREASED: Give the model time to learn continuous space
    lr_scheduler_type="cosine",          # ADDED: Smooth decay to 0
    warmup_ratio=0.1,                    # ADDED: Warmup phase for stability
    eval_strategy="epoch",
    save_strategy="epoch",
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