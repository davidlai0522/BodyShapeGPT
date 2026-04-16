"""
train_v4.py — Text-Conditioned Diffusion Model for SMPL Beta Parameters
========================================================================

Key change from v1-v3:
  v1-v3 : text → single point estimate  (MSE regression)
  v4    : text → P(beta | text)          (conditional DDPM)

Why this matters:
  "Tall athletic person" maps to many valid beta vectors. MSE forces the
  model to output their mean — a blurry body that isn't quite any of them.
  A diffusion model learns the full conditional distribution and can sample
  multiple plausible, diverse bodies from a vague description.

Architecture:
  1. Frozen LLM (Qwen2.5-0.5B)  — text encoder, embeddings pre-cached
  2. CosineSchedule (T=500)     — DDPM noise schedule
  3. BetaDenoiser (MLP, ~2M p)  — tiny denoiser over 10D space
  4. CFG dropout (p=0.1)        — enables classifier-free guidance at inference

Optimisations for RTX A5000 (24 GB, Ampere):
  1. Pre-cached embeddings  — Qwen runs ONCE; per training step is the
                              tiny denoiser only (~5-10x faster training)
  2. TF32 matmul            — free throughput on Ampere tensor cores
  3. BF16 autocast          — Ampere-native; no GradScaler needed
  4. torch.compile          — fused kernels on the denoiser MLP
  5. Batch size 1024        — denoiser fits many batches in 24 GB
  6. Cosine LR + warmup     — step-level; stable with large batch
  7. Batched DDIM eval      — vectorised over the whole test set at once

Inference:
  DDIM sampling (50 steps, deterministic) with guidance scale w.
  Text encoder is loaded only at inference (or interactively via GUI).
"""

import os
import ast
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Ampere: TF32 gives ~2x throughput on matmul/conv for free
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===========================================================================
# Config
# ===========================================================================
MODEL_NAME   = "Qwen/Qwen2.5-0.5B"
DATA_FILE    = "./BodyShapeGPT_dataset_1M.jsonl"
CACHE_FILE   = "./embeddings_cache.pt"      # pre-computed text embeddings
OUTPUT_DIR   = "./smpl_diffusion_v4_1"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

T            = 500       # diffusion timesteps — 500 is sufficient for 10D
BETA_DIM     = 10        # SMPL shape params
HIDDEN       = 512       # denoiser hidden size
T_DIM        = 128       # sinusoidal time embedding dim
MAX_SEQ_LEN  = 128

BATCH_SIZE       = 1024   # denoiser is tiny; A5000 handles this easily
N_EPOCHS         = 50     # each epoch is fast with pre-cached embeddings
LR               = 3e-4
WARMUP_STEPS     = 1000   # linear warmup before cosine decay
WEIGHT_DECAY     = 1e-4
CFG_DROPOUT      = 0.1
CFG_SCALE        = 2.0
GRAD_CLIP        = 1.0

CACHE_BATCH_SIZE  = 512   # batch size for the one-time embedding pass
EVAL_N_SAMPLES    = 5
MAX_EVAL_SAMPLES  = 2000
DDIM_STEPS        = 50
SAVE_EVERY        = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. Embedding cache builder
# ===========================================================================

def build_embedding_cache(
    data_file:   str = DATA_FILE,
    cache_file:  str = CACHE_FILE,
    model_name:  str = MODEL_NAME,
    max_length:  int = MAX_SEQ_LEN,
    batch_size:  int = CACHE_BATCH_SIZE,
    device:      str = DEVICE,
) -> None:
    """
    One-time pass: tokenise all texts, run Qwen, mean-pool, save to disk.

    Saves:
        cache_file  — {"text_emb": (N, D) float16,
                       "betas":    (N, 10) float32,
                       "text_dim": int,
                       "model_name": str}
    """
    if os.path.exists(cache_file):
        print(f"Embedding cache already exists: {cache_file}  (skipping)")
        return

    print(f"Building embedding cache → {cache_file}")
    print(f"  Loading data from {data_file} …")

    records = []
    with open(data_file) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    texts = [r["description"] for r in records]
    betas = torch.tensor(
        [ast.literal_eval(r["shape_params"]) for r in records], dtype=torch.float32
    )

    print(f"  Loaded {len(texts):,} samples. Loading text encoder …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    encoder.eval()

    all_embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  encoding"):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out  = encoder(input_ids=input_ids, attention_mask=attention_mask)
            h    = out.last_hidden_state                            # (B, L, D) fp16
            mask = attention_mask.unsqueeze(-1).half()             # (B, L, 1)
            emb  = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9) # (B, D) fp16
        all_embs.append(emb.cpu())

    text_emb = torch.cat(all_embs, dim=0)   # (N, D) float16
    text_dim = text_emb.shape[1]

    print(f"  Embedding shape: {tuple(text_emb.shape)}  dim={text_dim}")
    torch.save(
        {"text_emb": text_emb, "betas": betas,
         "text_dim": text_dim, "model_name": model_name},
        cache_file,
    )
    print(f"  Saved to {cache_file}")

    # Free VRAM before training starts
    del encoder
    torch.cuda.empty_cache()


# ===========================================================================
# 2. Dataset  (loads from cache — no Qwen at training time)
# ===========================================================================

class CachedSMPLDataset(Dataset):
    """
    Dataset backed by pre-computed text embeddings.
    __getitem__ returns (text_emb: float16, betas: float32).
    """
    def __init__(self, cache_file: str = CACHE_FILE):
        print(f"Loading embedding cache from {cache_file} …")
        data = torch.load(cache_file, map_location="cpu")
        self.text_embs = data["text_emb"]   # (N, D) float16
        self.betas     = data["betas"]       # (N, 10) float32
        self.text_dim  = int(data["text_dim"])
        print(f"  {len(self.betas):,} samples  |  text_dim={self.text_dim}")

    def __len__(self) -> int:
        return len(self.betas)

    def __getitem__(self, idx):
        return {
            "text_emb": self.text_embs[idx],   # (D,) float16
            "betas":    self.betas[idx],        # (10,) float32
        }


# ===========================================================================
# 3. Beta Normalizer  (standardize to ~N(0,1) per dimension)
# ===========================================================================

class BetaNormalizer:
    def __init__(self, betas: torch.Tensor):
        self.mean = betas.mean(0)
        self.std  = betas.std(0).clamp(min=1e-6)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def save(self, path: str):
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        d = torch.load(path, map_location="cpu")
        obj.mean = d["mean"]
        obj.std  = d["std"]
        return obj


# ===========================================================================
# 4. Cosine Noise Schedule
# ===========================================================================

class CosineSchedule(nn.Module):
    """
    Cosine beta schedule (Nichol & Dhariwal 2021).
    Better than linear for small-dimensional data — avoids too-noisy tails.
    """
    def __init__(self, T: int = T, s: float = 0.008):
        super().__init__()
        t = torch.linspace(0, T, T + 1)
        f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
        alphas_bar  = f / f[0]
        betas_sched = (1 - alphas_bar[1:] / alphas_bar[:-1]).clamp(1e-4, 0.999)
        alphas      = 1 - betas_sched
        alpha_bar   = torch.cumprod(alphas, 0)

        self.T = T
        self.register_buffer("alpha_bar",  alpha_bar)
        self.register_buffer("sqrt_ab",    alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab",  (1 - alpha_bar).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab   = self.sqrt_ab[t].unsqueeze(-1)
        s_1mab = self.sqrt_1mab[t].unsqueeze(-1)
        return s_ab * x0 + s_1mab * noise, noise


# ===========================================================================
# 5. Sinusoidal Time Embedding
# ===========================================================================

class SinPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([x.sin(), x.cos()], dim=-1)


# ===========================================================================
# 6. Denoiser MLP  (the only trained component)
# ===========================================================================

class BetaDenoiser(nn.Module):
    """
    Small MLP that predicts noise ε given (x_t, t, text_emb).
    null_text: learnable embedding used for CFG when text is dropped.
    """
    def __init__(self, text_dim: int, beta_dim: int = BETA_DIM,
                 hidden: int = HIDDEN, t_dim: int = T_DIM):
        super().__init__()

        self.t_mlp = nn.Sequential(
            SinPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.null_text = nn.Parameter(torch.zeros(1, text_dim))

        in_dim = beta_dim + t_dim + hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, beta_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(t)
        txt   = self.text_proj(text_emb)
        return self.net(torch.cat([x, t_emb, txt], dim=-1))


# ===========================================================================
# 7. Full Model
# ===========================================================================

class TextCondBetaDiffusion(nn.Module):
    def __init__(self, model_name: str, text_dim: int,
                 T: int = T, cfg_dropout: float = CFG_DROPOUT):
        super().__init__()
        self.cfg_dropout = cfg_dropout
        self._model_name = model_name

        # Frozen text encoder — loaded lazily at inference, not during training
        self._text_enc  = None
        self._tokenizer = None

        self.schedule  = CosineSchedule(T)
        self.denoiser  = BetaDenoiser(text_dim=text_dim)

    # ------------------------------------------------------------------
    # Lazy text encoder (inference only)
    # ------------------------------------------------------------------

    def _ensure_text_enc(self):
        if self._text_enc is None:
            print("Loading text encoder for inference …")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._text_enc = AutoModel.from_pretrained(
                self._model_name, torch_dtype=torch.float16
            ).to(next(self.denoiser.parameters()).device)
            for p in self._text_enc.parameters():
                p.requires_grad = False
            self._text_enc.eval()

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        self._ensure_text_enc()
        out  = self._text_enc(input_ids=input_ids, attention_mask=attention_mask)
        h    = out.last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    # ------------------------------------------------------------------
    # Training forward  (accepts pre-computed embeddings — no Qwen call)
    # ------------------------------------------------------------------

    def forward(self, betas: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        """
        betas    : (B, 10) normalised beta values
        text_emb : (B, D)  pre-cached float16 embeddings (cast to float32 inside)
        """
        B        = betas.shape[0]
        t        = torch.randint(0, self.schedule.T, (B,), device=betas.device)
        text_emb = text_emb.float()   # fp16 → fp32 for stable training

        if self.training and self.cfg_dropout > 0:
            drop     = torch.rand(B, device=betas.device) < self.cfg_dropout
            null     = self.denoiser.null_text.expand(B, -1)
            text_emb = torch.where(drop.unsqueeze(-1), null, text_emb)

        x_noisy, noise = self.schedule.q_sample(betas, t)
        pred_noise     = self.denoiser(x_noisy, t, text_emb)
        return F.mse_loss(pred_noise, noise)

    # ------------------------------------------------------------------
    # Batched DDIM sampling from pre-computed embeddings (eval / GUI)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_from_emb(self, text_emb: torch.Tensor,
                        n_samples: int = 1,
                        ddim_steps: int = DDIM_STEPS,
                        cfg_scale: float = CFG_SCALE) -> torch.Tensor:
        """
        text_emb : (B, D)  float16 or float32
        Returns  : (B, n_samples, BETA_DIM)  in normalised space
        """
        device   = text_emb.device
        B        = text_emb.shape[0]
        text_emb = text_emb.float()

        # Expand each description for n_samples
        t_rep    = text_emb.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        null_rep = self.denoiser.null_text.expand(B * n_samples, -1)

        step = self.schedule.T // ddim_steps
        ts   = list(range(step - 1, self.schedule.T, step))[::-1]
        x    = torch.randn(B * n_samples, BETA_DIM, device=device)

        for idx, t_val in enumerate(ts):
            t  = torch.full((B * n_samples,), t_val, device=device, dtype=torch.long)
            ab = self.schedule.alpha_bar[t_val]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eps_text = self.denoiser(x, t, t_rep)
                if cfg_scale != 1.0:
                    eps_null = self.denoiser(x, t, null_rep)
                    eps = eps_null + cfg_scale * (eps_text - eps_null)
                else:
                    eps = eps_text

            eps     = eps.float()
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-6)
            x0_pred = x0_pred.clamp(-5, 5)

            if idx < len(ts) - 1:
                ab_prev = self.schedule.alpha_bar[ts[idx + 1]]
                x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
            else:
                x = x0_pred

        return x.reshape(B, n_samples, BETA_DIM)

    # ------------------------------------------------------------------
    # Interactive DDIM sampling (GUI / load_model_for_inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               n_samples: int = 1, ddim_steps: int = DDIM_STEPS,
               cfg_scale: float = CFG_SCALE) -> torch.Tensor:
        """
        Encode text on-the-fly and sample.
        Returns (n_samples, BETA_DIM) in normalised space.
        """
        text_emb = self.encode_text(input_ids, attention_mask)   # (1, D)
        text_emb = text_emb.repeat(n_samples, 1)                  # (n_samples, D)
        null_emb = self.denoiser.null_text.expand(n_samples, -1)

        step = self.schedule.T // ddim_steps
        ts   = list(range(step - 1, self.schedule.T, step))[::-1]
        x    = torch.randn(n_samples, BETA_DIM, device=input_ids.device)

        for idx, t_val in enumerate(ts):
            t  = torch.full((n_samples,), t_val, device=x.device, dtype=torch.long)
            ab = self.schedule.alpha_bar[t_val]

            eps_text = self.denoiser(x, t, text_emb)
            if cfg_scale != 1.0:
                eps_null = self.denoiser(x, t, null_emb)
                eps = eps_null + cfg_scale * (eps_text - eps_null)
            else:
                eps = eps_text

            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-6)
            x0_pred = x0_pred.clamp(-5, 5)

            if idx < len(ts) - 1:
                ab_prev = self.schedule.alpha_bar[ts[idx + 1]]
                x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
            else:
                x = x0_pred

        return x


# ===========================================================================
# 8. LR schedule helper  (linear warmup + cosine decay, step-level)
# ===========================================================================

def _get_lr_lambda(warmup_steps: int, total_steps: int,
                   min_ratio: float = 0.01):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_ratio, cosine)
    return lr_lambda


# ===========================================================================
# 9. Evaluation  (batched — no per-sample Python loop)
# ===========================================================================

@torch.no_grad()
def evaluate(model: TextCondBetaDiffusion, loader: DataLoader,
             normalizer: BetaNormalizer, device: str) -> dict:
    """
    Vectorised eval: sample EVAL_N_SAMPLES bodies per description,
    report min-MSE and mean-MSE in normalised space.
    """
    model.eval()
    min_mses: list[float] = []
    mean_mses: list[float] = []

    for batch in loader:
        text_emb = batch["text_emb"].to(device)            # (B, D)
        gt_norm  = normalizer.normalize(batch["betas"].to(device))  # (B, 10)

        # (B, N, 10)
        samples = model.sample_from_emb(text_emb, n_samples=EVAL_N_SAMPLES)

        gt_exp  = gt_norm.unsqueeze(1)                     # (B, 1, 10)
        mses    = ((samples - gt_exp) ** 2).mean(-1)       # (B, N)

        min_mses.extend(mses.min(dim=1).values.cpu().tolist())
        mean_mses.extend(mses.mean(dim=1).cpu().tolist())

    return {
        "min_mse_norm":  float(np.mean(min_mses)),
        "mean_mse_norm": float(np.mean(mean_mses)),
    }


# ===========================================================================
# 10. Checkpoint helpers
# ===========================================================================

def _unwrap_compiled(module: nn.Module) -> nn.Module:
    """Return the original module from a torch.compile wrapper, if present."""
    return getattr(module, "_orig_mod", module)


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """
    torch.compile saves weights with an '_orig_mod.' prefix.
    Strip it so the state dict loads cleanly into an uncompiled module.
    """
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {
            k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }
    return state_dict


# ===========================================================================
# 11. Main — Training Loop
# ===========================================================================

def main():
    print(f"Device: {DEVICE}")

    # ------------------------------------------------------------------
    # Step 1: build embedding cache (no-op if already exists)
    # ------------------------------------------------------------------
    build_embedding_cache()

    # ------------------------------------------------------------------
    # Step 2: dataset + split
    # ------------------------------------------------------------------
    dataset = CachedSMPLDataset(CACHE_FILE)

    n_test  = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    if len(test_ds) > MAX_EVAL_SAMPLES:
        test_ds, _ = random_split(
            test_ds,
            [MAX_EVAL_SAMPLES, len(test_ds) - MAX_EVAL_SAMPLES],
            generator=torch.Generator().manual_seed(42),
        )

    # ------------------------------------------------------------------
    # Step 3: normalizer (fit on training betas only)
    # ------------------------------------------------------------------
    train_betas = dataset.betas[train_ds.indices]
    normalizer  = BetaNormalizer(train_betas)
    normalizer.save(os.path.join(OUTPUT_DIR, "normalizer.pt"))
    print(f"Beta mean : {normalizer.mean.numpy().round(3)}")
    print(f"Beta std  : {normalizer.std.numpy().round(3)}")

    dataset.betas = normalizer.normalize(dataset.betas)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    print(f"Train: {n_train:,}  |  Test (capped): {len(test_ds):,}")

    # ------------------------------------------------------------------
    # Step 4: model
    # ------------------------------------------------------------------
    model = TextCondBetaDiffusion(
        model_name=MODEL_NAME,
        text_dim=dataset.text_dim,
    ).to(DEVICE)

    # torch.compile fuses denoiser kernels on Ampere (PyTorch ≥ 2.0).
    # "default" mode: TorchInductor fusion without CUDA graphs.
    # "reduce-overhead" is avoided because CUDA graphs reuse output buffers,
    # which causes a RuntimeError when the denoiser is called twice per step
    # (eps_text + eps_null for CFG) — the second call overwrites the first.
    if hasattr(torch, "compile"):
        print("Compiling denoiser with torch.compile …")
        model.denoiser = torch.compile(model.denoiser, mode="default")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # ------------------------------------------------------------------
    # Step 5: optimizer + step-level LR schedule
    # ------------------------------------------------------------------
    optimizer    = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    total_steps  = N_EPOCHS * len(train_loader)
    lr_scheduler = LambdaLR(optimizer, _get_lr_lambda(WARMUP_STEPS, total_steps))

    print(f"Total steps: {total_steps:,}  |  Warmup: {WARMUP_STEPS}")

    # ------------------------------------------------------------------
    # Step 6: training loop  (BF16 autocast, no GradScaler needed)
    # ------------------------------------------------------------------
    best_min_mse = float("inf")
    log_path     = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,min_mse_norm,mean_mse_norm,lr\n")

    global_step = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            text_emb = batch["text_emb"].to(DEVICE)   # (B, D) fp16
            betas    = batch["betas"].to(DEVICE)        # (B, 10) fp32

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(betas, text_emb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

        avg_loss   = epoch_loss / len(train_loader)
        current_lr = lr_scheduler.get_last_lr()[0]

        metrics  = evaluate(model, test_loader, normalizer, DEVICE)
        min_mse  = metrics["min_mse_norm"]
        mean_mse = metrics["mean_mse_norm"]

        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | "
            f"loss={avg_loss:.4f} | "
            f"min_mse={min_mse:.4f} | "
            f"mean_mse={mean_mse:.4f} | "
            f"lr={current_lr:.2e}"
        )

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{min_mse:.6f},{mean_mse:.6f},{current_lr:.2e}\n")

        if min_mse < best_min_mse:
            best_min_mse = min_mse
            torch.save(
                {
                    "epoch":         epoch,
                    "denoiser":      _unwrap_compiled(model.denoiser).state_dict(),
                    "schedule":      model.schedule.state_dict(),
                    "text_dim":      dataset.text_dim,
                    "min_mse_norm":  min_mse,
                    "mean_mse_norm": mean_mse,
                },
                os.path.join(OUTPUT_DIR, "best_model.pt"),
            )
            print(f"  ↳ Saved best model (min_mse={min_mse:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save(
                {"epoch": epoch,
                 "denoiser": _unwrap_compiled(model.denoiser).state_dict(),
                 "text_dim": dataset.text_dim},
                os.path.join(OUTPUT_DIR, f"checkpoint_ep{epoch:03d}.pt"),
            )

    print(f"\nTraining complete. Best min_mse_norm={best_min_mse:.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR}/")


# ===========================================================================
# 11. Inference helper (import from other scripts / GUI)
# ===========================================================================

def load_model_for_inference(
    checkpoint_dir: str   = OUTPUT_DIR,
    cfg_scale:      float = CFG_SCALE,
    ddim_steps:     int   = DDIM_STEPS,
):
    """
    Load the trained diffusion model and normalizer.

    Usage:
        model, normalizer, tokenizer, sample_fn = load_model_for_inference()
        betas = sample_fn("Tall person with broad shoulders", n_samples=5)
        # betas: numpy array (5, 10) in real SMPL units
    """
    ckpt = torch.load(
        os.path.join(checkpoint_dir, "best_model.pt"), map_location=DEVICE
    )
    text_dim = int(ckpt["text_dim"])

    model = TextCondBetaDiffusion(
        model_name=MODEL_NAME, text_dim=text_dim
    ).to(DEVICE)
    model.denoiser.load_state_dict(_strip_compiled_prefix(ckpt["denoiser"]))
    model.schedule.load_state_dict(ckpt["schedule"])
    model.eval()

    normalizer = BetaNormalizer.load(os.path.join(checkpoint_dir, "normalizer.pt"))

    # Trigger lazy load of text encoder + tokenizer
    model._ensure_text_enc()
    tokenizer = model._tokenizer

    def sample_fn(description: str, n_samples: int = 1) -> np.ndarray:
        enc = tokenizer(
            description, max_length=MAX_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            samples_norm = model.sample(
                input_ids, attention_mask,
                n_samples=n_samples, ddim_steps=ddim_steps, cfg_scale=cfg_scale,
            )
            samples = normalizer.denormalize(samples_norm)

        return samples.cpu().numpy()

    return model, normalizer, tokenizer, sample_fn


if __name__ == "__main__":
    main()
