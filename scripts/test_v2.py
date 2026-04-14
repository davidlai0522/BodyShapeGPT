import torch
import torch.nn as nn
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
MODEL_NAME = "Qwen/Qwen2.5-3B"
CHECKPOINT_PATH = "/home/schaeffler-pte-ltd/david_ws/BodyShapeGPT/smpl_regressor_checkpoints_v2/final_smpl_regressor.pth"
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
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # must match training
    lora_dropout=0.05,
    bias="none",
)
base_llm = get_peft_model(base_llm, lora_config)

# Initialize full model
hidden_size = base_llm.config.hidden_size
model = LLMToSMPLRegressor(base_llm, hidden_size)

# Load the saved state dictionary (contains both LoRA weights and MLP weights)
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
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
# 4. Test It
# ==========================================
if __name__ == "__main__":
    test_descriptions = [
        "Person with an average height, tall neck, long arms, and broad shoulders.",
        "A very tall, highly muscular individual with a heavy build and thick legs.",
        "A petite frame with narrow shoulders, short stature, and low body mass."
    ]
    
    print("\n--- Running Inference ---")
    for desc in test_descriptions:
        print(f"\nInput: {desc}")
        betas = predict_smpl_shape(desc)
        # Formatting to 3 decimal places for readability
        formatted_betas = [f"{b:.3f}" for b in betas]
        print(f"Output Betas: {formatted_betas}")