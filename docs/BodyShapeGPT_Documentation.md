# BodyShapeGPT — Technical Documentation

> Reference document for `generate_dataset.py` and `train_v4.py`

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataset Generation — `generate_dataset.py`](#2-dataset-generation--generate_datasetpy)
   - 2.1 [Pipeline Overview](#21-pipeline-overview)
   - 2.2 [SMPL Forward Pass & Measurement Extraction](#22-smpl-forward-pass--measurement-extraction)
   - 2.3 [Threshold Calibration & Categorisation](#23-threshold-calibration--categorisation)
   - 2.4 [Body Shape Classification](#24-body-shape-classification)
   - 2.5 [Description Generation](#25-description-generation)
   - 2.6 [Optional LLM Rephrasing](#26-optional-llm-rephrasing)
   - 2.7 [CLI Reference](#27-cli-reference)
3. [Why a Probabilistic Model?](#3-why-a-probabilistic-model)
   - 3.1 [The Problem with Regression](#31-the-problem-with-regression)
   - 3.2 [Why Diffusion Models Solve This](#32-why-diffusion-models-solve-this)
   - 3.3 [Iteration History — v1 to v4](#33-iteration-history--v1-to-v4)
4. [Architecture — `train_v4.py`](#4-architecture--train_v4py)
   - 4.1 [System Architecture Diagram](#41-system-architecture-diagram)
   - 4.2 [Text Encoder — Qwen2.5-0.5B](#42-text-encoder--qwen250b)
   - 4.3 [Cosine Noise Schedule](#43-cosine-noise-schedule)
   - 4.4 [Sinusoidal Time Embedding](#44-sinusoidal-time-embedding)
   - 4.5 [BetaDenoiser MLP](#45-betadenoiser-mlp)
   - 4.6 [Classifier-Free Guidance (CFG)](#46-classifier-free-guidance-cfg)
   - 4.7 [DDIM Sampling](#47-ddim-sampling)
   - 4.8 [BetaNormalizer](#48-betanormalizer)
5. [Training Procedure](#5-training-procedure)
   - 5.1 [Embedding Cache](#51-embedding-cache)
   - 5.2 [Training Loop](#52-training-loop)
   - 5.3 [Evaluation](#53-evaluation)
   - 5.4 [Configuration Reference](#54-configuration-reference)
   - 5.5 [Saved Artefacts](#55-saved-artefacts)
6. [Inference](#6-inference)
7. [Hardware & Performance](#7-hardware--performance)
8. [Paper vs This Implementation](#8-paper-vs-this-implementation)
   - 8.1 [What the Paper Actually Does](#81-what-the-paper-actually-does)
   - 8.2 [Difference-by-Difference Breakdown](#82-difference-by-difference-breakdown)
   - 8.3 [Genuine Innovations](#83-genuine-innovations)
   - 8.4 [Where the Paper Has an Advantage](#84-where-the-paper-has-an-advantage)
   - 8.5 [Summary](#85-summary)

---

## 1. System Overview

BodyShapeGPT converts a free-text body description into SMPL shape parameters (β₀ – β₉). These 10 values fully determine a body's height, weight distribution, limb proportions, and overall silhouette when passed to the SMPL parametric body model.

```
Free-text description
        │
        ▼
  Qwen2.5-0.5B          ← frozen text encoder, mean-pooled output
        │  (896-dim embedding)
        ▼
  BetaDenoiser (MLP)     ← conditional DDPM, samples P(β | text)
        │  (β ∈ ℝ¹⁰, normalised)
        ▼
  BetaNormalizer         ← denormalise to real SMPL units
        │
        ▼
  SMPL forward pass      → 3-D mesh (6 890 vertices, 13 776 faces)
```

**Why 10 dimensions?**
SMPL encodes body shape as the first 10 principal components of a PCA learned from thousands of body scans. These 10 coefficients capture the vast majority of realistic human variation (height, weight, limb length, torso width, etc.). The task is therefore a 10-dimensional conditional generation problem rather than image generation, which is why a small MLP denoiser is sufficient.

**End-to-end workflow:**

```
1. generate_dataset.py  →  BodyShapeGPT_dataset_1M.jsonl   (1M SMPL + description pairs)
2. train_v4.py          →  embeddings_cache.pt + best_model.pt
3. Inference            →  sample_fn("description") → β array (5, 10)
```

---

## 2. Dataset Generation — `generate_dataset.py`

### 2.1 Pipeline Overview

```
Sample betas ~ N(0, I₁₀), clip to [−3.5, 3.5]
        │
        ▼
SMPL forward pass (batched)  →  vertices (6890×3), joints (45×3)
        │
        ▼
extract_measurements()       →  12 geometric measurements (metres)
        │
        ▼
calibrate_thresholds()       →  20/40/60/80th percentiles per measurement
        │
        ▼
measurements_to_categories() →  {attr: very_low|low|average|high|very_high}
        │
        ├── classify_body_shape()  →  hourglass|pear|apple|rectangle|inverted_triangle
        │
        ▼
_generate_description()      →  natural-language sentence
        │
        ├── (optional) LLMRephraser  →  paraphrase with Qwen2.5-1.5B-Instruct
        │
        ▼
JSONL record: {"description": "…", "shape_params": "[β₀, …, β₉]"}
```

Accuracy guarantees:
- Measurements are derived directly from SMPL geometry — no estimation or proxy.
- Thresholds are calibrated from the same β distribution used for sampling, so category assignments are statistically correct by construction.
- When LLM rephrasing is used, a keyword validation step rejects any output that changes an attribute level, falling back to the template version.

---

### 2.2 SMPL Forward Pass & Measurement Extraction

All samples are run through SMPL at zero pose (T-pose) so measurements are purely shape-driven, not pose-dependent.

**`extract_measurements(verts, joints)`** returns 12 values (all in metres):

| Measurement | Method |
|---|---|
| `height` | `verts[:, 1].max() − verts[:, 1].min()` |
| `neck_length` | Joint distance: `neck → head` |
| `arm_length` | Chain: `l_shoulder → l_elbow → l_wrist` |
| `leg_length` | Chain: `l_hip → l_knee → l_ankle` |
| `shoulder_width` | Joint distance: `l_shoulder → r_shoulder` (biacromial) |
| `hip_width` | X-span of vertices near hip-joint height, clipped to `|x| ≤ 0.35` |
| `waist_width` | X-span of vertices near `spine2` height, clipped to `|x| ≤ 0.30` (shape only) |
| `waist_depth` | Front-to-back Z-span at `spine2` height, arms excluded `|x| ≤ 0.24` |
| `chest_depth` | Front-to-back Z-span at `spine3` height, arms excluded |
| `arm_girth` | Cross-sectional extent of vertices near `l_elbow` |
| `leg_girth` | Cross-sectional extent of vertices near `l_knee` |
| `bmi_proxy` | Torso voxel volume / height², T-pose arms excluded (`|x| ≤ 0.28`) |

`waist_width` is used only for body shape classification; it is not exposed in descriptions.

---

### 2.3 Threshold Calibration & Categorisation

`calibrate_thresholds()` runs SMPL on 5 000 random bodies and computes the 20th, 40th, 60th, and 80th percentile of each measurement. These four values divide the distribution into five equal-frequency bins:

| Bin | Range | Label |
|---|---|---|
| 0–20th pct | `value < p20` | `very_low` |
| 20–40th pct | `p20 ≤ value < p40` | `low` |
| 40–60th pct | `p40 ≤ value < p60` | `average` |
| 60–80th pct | `p60 ≤ value < p80` | `high` |
| 80–100th pct | `value ≥ p80` | `very_high` |

Because thresholds are percentile-based, each category naturally contains ~20% of all generated bodies, so the dataset is balanced across all five levels for every attribute.

---

### 2.4 Body Shape Classification

`classify_body_shape()` assigns one of five canonical body types, evaluated in priority order:

```
1. apple            — bmi_proxy ∈ {high, very_high}  AND  waist_depth ∈ {high, very_high}
2. inverted_triangle — shoulder_width / hip_width > 1.15
3. pear              — shoulder_width / hip_width < 0.87
4. hourglass         — waist_width / hip_width < 0.80  OR  waist_depth ∈ {very_low, low}
5. rectangle         — everything else
```

The ratios use `waist_width` (lateral measurement at spine2) rather than `waist_depth` for more geometrically accurate proportions.

---

### 2.5 Description Generation

**`_generate_description(cats, body_shape)`** composes one natural-language sentence by:

1. Always including a height phrase (e.g. *"tall"*, *"very short"*).
2. Sampling 1–3 additional attributes via `_select_attrs()`, biased toward non-average values (weight 3.0 for `very_low`/`very_high`, 1.5 for `low`/`high`, 0.6 for `average`).
3. Selecting a random synonym from the `ADJ` vocabulary for each attribute and category.
4. With 65% probability, including a body shape phrase (e.g. *"an hourglass figure"*) from `SHAPE_PHRASES`.
5. Filling one of 9 shape-aware or 8 plain sentence templates, chosen at random.

This produces varied, concise descriptions without needing an LLM:

> *"A tall person with an hourglass figure and broad shoulders."*
> *"Very short individual featuring a pear-shaped figure and a slim build."*
> *"Short person with a pear body type — narrow shoulders."*

**Vocabulary coverage:** The `ADJ` dict provides 3–9 synonym phrases per (attribute, level) pair, covering neutral measurement language and colloquial terms for BMI (`fat build`, `skinny`, `heavyset`, `lean`, etc.).

---

### 2.6 Optional LLM Rephrasing

`LLMRephraser` (enabled with `--rephrase`) uses Qwen2.5-1.5B-Instruct to paraphrase each template sentence, adding more lexical diversity. A validation step checks that key attribute keywords are preserved:

```
prompt: "Rephrase: '<template sentence>'"
        system: "Keep ALL attributes and levels identical. Output ONE sentence."

→ validate that height & BMI keywords survive → keep rephrased
                                                → else keep original
```

This ensures accuracy is never sacrificed for variety.

---

### 2.7 CLI Reference

```
python generate_dataset.py [OPTIONS]

  --n            INT    Samples to generate              [default: 21000]
  --output       PATH   Output JSONL file                [default: BodyShapeGPT_dataset_new.jsonl]
  --gender       STR    neutral | male | female          [default: neutral]
  --batch        INT    SMPL batch size                  [default: 128]
  --n-calib      INT    Bodies for threshold calibration [default: 5000]
  --seed         INT    Random seed                      [default: 42]
  --rephrase            Enable LLM rephrasing pass
  --rephrase-model STR  HuggingFace model for rephrasing [default: Qwen/Qwen2.5-1.5B-Instruct]
  --validate            Audit an existing JSONL file instead of generating
  --validate-n   INT    Samples to audit                 [default: 200]
```

**Validation mode** (`--validate`) re-runs SMPL on a random sample from the file, re-extracts measurements, and reports per-attribute accuracy for height, BMI, and body shape.

---

## 3. Why a Probabilistic Model?

### 3.1 The Problem with Regression

The naive approach (v1–v3) trains an MLP to regress a single β vector from a text embedding using MSE loss:

```
text → encoder → MLP → β̂     loss = ‖β̂ − β_true‖²
```

This fails for a fundamental reason: **natural language descriptions of bodies are inherently ambiguous**.

> *"A tall athletic person"* — this describes a continuous family of bodies.
> Heights can range 185–200 cm, shoulder widths can vary, limb proportions differ.
> There is no single "correct" β vector for this description.

When trained with MSE, the model is forced to output the *mean* of all valid bodies for that description. This mean body exists nowhere in the real distribution — it is an average that is not quite any real person. The result is systematically bland, mid-range predictions regardless of the input.

Additionally, regression produces **zero diversity** — the same description always produces the same output, making it impossible to explore the space of plausible bodies.

### 3.2 Why Diffusion Models Solve This

A conditional diffusion model learns the full distribution **P(β | text)** rather than a point estimate. It iteratively denoises a sample of Gaussian noise into a plausible β vector, guided by the text embedding:

```
Regression:  text → single β̂           (collapses distribution to mean)
Diffusion:   text → P(β | text) → sample β   (draws from the distribution)
```

This directly solves both problems:

1. **Ambiguity is preserved, not lost.** The model can draw multiple different but equally valid bodies from the same vague description.
2. **Diversity is tunable.** Classifier-Free Guidance (CFG) lets you dial between maximum diversity and strict text adherence at inference time, without retraining.
3. **Rare extremes are not averaged away.** A regression model trained on a body described as *"extremely thin with long arms"* will average this against similar but less extreme bodies. A diffusion model can sample the extreme directly.

The cost is inference speed (50 DDIM steps vs a single forward pass), but for a 10-dimensional space this is negligible — generation takes milliseconds.

### 3.3 Iteration History — v1 to v4

| Version | Approach | Key Problem |
|---|---|---|
| v1 | MSE regression, Qwen2.5-3B, last-token pooling | Bugs: fp16 overflow, tensor serialisation errors, narrow LoRA |
| v2 | MSE regression, 3B, mean pooling, expanded LoRA | Still a point estimate; gradient instability with fp16 |
| v3 | MSE regression, 0.5B, bf16, L2 regularisation | Fundamental ambiguity problem remains; early stopping hides it |
| **v4** | **Conditional DDPM, frozen encoder, pre-cached embeddings** | **Solves ambiguity; 250× fewer trainable params; 10× faster training** |

---

## 4. Architecture — `train_v4.py`

### 4.1 System Architecture Diagram

```
┌─────────────────────── Training ──────────────────────────────────┐
│                                                                    │
│  embeddings_cache.pt                                               │
│  ┌─────────────────────────────────────────────────┐              │
│  │  Qwen2.5-0.5B (frozen)                          │              │
│  │  input_ids (B, 128) → last_hidden (B, 128, 896) │ (one-time)   │
│  │  masked mean-pool → text_emb (N, 896) fp16      │              │
│  └─────────────────────────────────────────────────┘              │
│                │                                                   │
│                ▼  (loaded each batch, no Qwen at train time)       │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  BetaDenoiser (~2M params, only trainable component)         │ │
│  │                                                              │ │
│  │  Input:  [x_t (10) | t_emb (128) | text_proj (512)] → (650) │ │
│  │                                                              │ │
│  │  Linear(650→512) → SiLU → LayerNorm                         │ │
│  │  Linear(512→512) → SiLU → LayerNorm                         │ │
│  │  Linear(512→256) → SiLU                                      │ │
│  │  Linear(256→ 10)                 → predicted noise ε̂        │ │
│  │                                                              │ │
│  │  loss = MSE(ε̂, ε)                                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────── Inference ─────────────────────────────────┐
│                                                                    │
│  text description                                                  │
│        │                                                           │
│        ▼  (Qwen lazy-loaded once)                                  │
│  text_emb (896)                                                    │
│        │                                                           │
│        ▼  DDIM (50 steps)                                          │
│  x_T ~ N(0, I₁₀)  →  BetaDenoiser + CFG  →  β̂_norm (10)         │
│        │                                                           │
│        ▼  BetaNormalizer.denormalise()                             │
│  β (10) in real SMPL units                                         │
└────────────────────────────────────────────────────────────────────┘
```

---

### 4.2 Text Encoder — Qwen2.5-0.5B

Qwen2.5-0.5B is used as a **frozen** text encoder. It is never fine-tuned. Its role is solely to convert a variable-length description into a fixed-size 896-dimensional embedding via masked mean-pooling over the final hidden layer:

```
hidden_state : (B, L, 896)   ← Qwen last transformer layer
mask         : (B, L, 1)     ← attention mask (0 for padding tokens)

embedding = Σ(hidden × mask, dim=L) / Σ(mask, dim=L)   →   (B, 896)
```

Mean-pooling is preferred over last-token pooling because it aggregates information from all non-padding tokens, producing a more stable and content-rich sentence representation.

**Why Qwen2.5-0.5B specifically?**
- 0.5B parameters: small enough to run on consumer hardware, large enough to have rich semantic representations
- 896-dim hidden size: sufficient conditioning signal for a 10-dimensional target space
- Handles body description vocabulary (anatomical terms, size adjectives, shape types) well

**Embedding cache:** All 1M embeddings are computed once and stored as a `(1M, 896)` float16 tensor (~1.8 GB). Training only ever loads from this cache, so Qwen is never instantiated during the training loop.

---

### 4.3 Cosine Noise Schedule

The forward diffusion process progressively corrupts a clean β into noise over `T = 500` timesteps using the cosine schedule (Nichol & Dhariwal, 2021):

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

The noisy sample at timestep $t$ is:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

**Why cosine over linear?**
The linear schedule degrades quality at the start (too noisy too quickly) and end (near-zero signal). The cosine schedule decays more gradually, which is especially important for low-dimensional data like 10D beta vectors where each dimension carries significant information.

**Why `T = 500` instead of 1000?**
A 10-dimensional Gaussian is much faster to corrupt than a 64×64 image. `T = 500` reaches near-complete noise while halving per-epoch denoiser calls.

---

### 4.4 Sinusoidal Time Embedding

The denoiser needs to know "how noisy" the current sample is. The scalar timestep `t` is mapped to a `T_DIM = 128`-dimensional continuous vector using sinusoidal positional encoding:

$$\text{emb}_{2i} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{emb}_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

This is then passed through a 2-layer MLP (`128 → 256 → 128`) to allow the network to learn a non-linear transformation of the time signal.

---

### 4.5 BetaDenoiser MLP

The only component with **trainable parameters (~2M)**. Given a noisy beta `x_t`, timestep `t`, and text embedding, it predicts the noise `ε` that was added:

**Input assembly:**

| Input | Shape | Description |
|---|---|---|
| `x_t` | `(B, 10)` | Noisy beta at timestep `t` |
| `t_emb` | `(B, 128)` | Sinusoidal time embedding → 2-layer MLP |
| `text_proj` | `(B, 512)` | text_emb `(896)` → 2-layer MLP projection |
| **concat** | **(B, 650)** | **Concatenated input to main network** |

**Network:**

```
Linear(650 → 512) → SiLU → LayerNorm
Linear(512 → 512) → SiLU → LayerNorm
Linear(512 → 256) → SiLU
Linear(256 →  10)                      ← predicted noise ε̂
```

**Design choices:**
- `SiLU` (Swish) activation — smoother than ReLU; empirically better for continuous regression tasks
- `LayerNorm` after each large layer — stabilises training, especially with bf16 mixed precision
- Separate MLPs for time and text projections — allows each modality's signal to be scaled independently before concatenation
- `null_text` — a learnable `(1, 896)` parameter used in place of the text embedding during CFG dropout. Allows the denoiser to learn unconditional generation, enabling CFG at inference.

---

### 4.6 Classifier-Free Guidance (CFG)

CFG adds a post-training control knob that adjusts how strongly the output is steered by the text description.

**Training:** with probability `CFG_DROPOUT = 0.10`, the text embedding is replaced by the learnable `null_text` token. This trains the denoiser on both conditional (`text_emb`) and unconditional (`null_text`) input simultaneously.

**Inference:** the denoiser is run twice per timestep — once with the real text embedding and once with `null_text`. The outputs are blended:

$$\hat{\varepsilon} = \hat{\varepsilon}_{\text{uncond}} + w \cdot (\hat{\varepsilon}_{\text{text}} - \hat{\varepsilon}_{\text{uncond}})$$

where `w = CFG_SCALE` (default 2.0).

| Scale `w` | Behaviour |
|---|---|
| 1.0 | Unconditional — ignores text, maximum diversity |
| 2.0 | Balanced — recommended default |
| 4.0+ | Strict text adherence, reduced diversity |

---

### 4.7 DDIM Sampling

At inference, Denoising Diffusion Implicit Models (DDIM, Song et al. 2020) are used instead of standard DDPM. DDIM is deterministic and skips most timesteps, requiring only 50 steps (vs. 500 in training) with minimal quality loss.

```
For t = T, T−Δ, …, 0:
  1. Predict noise:    ε̂ = denoiser(x_t, t, text_emb)  [+ CFG blend]
  2. Estimate x̂₀:     x̂₀ = (x_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t
  3. Re-noise:         x_{t−Δ} = √ᾱ_{t−Δ} · x̂₀ + √(1−ᾱ_{t−Δ}) · ε̂

Return x̂₀ at the final step
```

**Why DDIM over DDPM at inference?**
DDPM requires a sample at every timestep (500 forward passes). DDIM skips timesteps with a deterministic rule — 50 passes suffice, giving a 10× speedup with negligible quality difference at this dimensionality.

---

### 4.8 BetaNormalizer

Raw SMPL betas have different scales per dimension (some span [−1, 1], others [−3, 3]). Training a diffusion model on unnormalised data would bias the noise schedule. `BetaNormalizer` standardises each dimension independently:

$$\hat{x} = \frac{x - \mu}{\sigma}, \quad \mu, \sigma \text{ computed from training betas only}$$

This ensures the diffusion model operates in a well-conditioned `~N(0,1)` space. At inference, outputs are denormalised before passing to SMPL.

---

## 5. Training Procedure

### 5.1 Embedding Cache

The first time `train_v4.py` runs, `build_embedding_cache()` processes all 1M descriptions through Qwen and saves the result:

```
Qwen2.5-0.5B → mean-pool → (1 000 000, 896) float16 → embeddings_cache.pt
```

This takes ~20–30 min on an A5000 and is skipped on all subsequent runs. The cache contains `{text_emb, betas, text_dim, model_name}`.

### 5.2 Training Loop

```python
# Step-level cosine LR with linear warmup
# Steps 0 → 1000 : linear 0 → 3e-4
# Steps 1000 → end : cosine 3e-4 → 3e-6

for batch in train_loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(betas_norm, text_emb)   # noise-prediction MSE
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
```

**BFloat16 autocast** is used (not float16). BFloat16 has the same 8-bit exponent as float32, so it cannot overflow — no GradScaler is needed. This was the root cause of the `grad_norm=inf` spikes in v2/v3.

**Batch size 1024** — the denoiser is tiny (~2M params) so a large batch fits easily and stabilises gradient estimates.

### 5.3 Evaluation

After each epoch, `evaluate()` draws `EVAL_N_SAMPLES = 5` beta vectors per test description via batched DDIM and reports:

- `min_mse_norm` — MSE of the closest sample to ground truth (oracle upper bound on quality)
- `mean_mse_norm` — average MSE across all samples

The best checkpoint (lowest `min_mse_norm`) is saved to `best_model.pt`. The test set is capped at 2 000 samples to prevent evaluation from dominating wall-clock time.

### 5.4 Configuration Reference

| Constant | Value | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B` | Frozen text encoder |
| `T` | `500` | Diffusion timesteps |
| `BETA_DIM` | `10` | SMPL shape parameter count |
| `HIDDEN` | `512` | Denoiser hidden layer width |
| `T_DIM` | `128` | Sinusoidal time embedding dimension |
| `MAX_SEQ_LEN` | `128` | Token length cap for text encoder |
| `BATCH_SIZE` | `1024` | Training batch size |
| `N_EPOCHS` | `50` | Training epochs |
| `LR` | `3e-4` | Peak learning rate (AdamW) |
| `WARMUP_STEPS` | `1000` | Linear LR warmup steps |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `CFG_DROPOUT` | `0.10` | Probability of replacing text with null token during training |
| `CFG_SCALE` | `2.0` | Default guidance scale at inference |
| `GRAD_CLIP` | `1.0` | Gradient norm clip |
| `EVAL_N_SAMPLES` | `5` | Samples drawn per test description during eval |
| `MAX_EVAL_SAMPLES` | `2000` | Test set cap |
| `DDIM_STEPS` | `50` | Denoising steps at inference |
| `SAVE_EVERY` | `5` | Periodic checkpoint frequency (epochs) |

### 5.5 Saved Artefacts

| File | Contents |
|---|---|
| `embeddings_cache.pt` | `{text_emb, betas, text_dim, model_name}` |
| `best_model.pt` | `{epoch, denoiser, schedule, text_dim, min_mse_norm, mean_mse_norm}` |
| `checkpoint_ep###.pt` | Periodic snapshot every 5 epochs |
| `normalizer.pt` | `{mean, std}` for beta standardisation |
| `training_log.csv` | `epoch, train_loss, min_mse_norm, mean_mse_norm, lr` |

---

## 6. Inference

```python
from train_v4 import load_model_for_inference

model, normalizer, tokenizer, sample_fn = load_model_for_inference(
    checkpoint_dir="./smpl_diffusion_v4",
    cfg_scale=2.0,
    ddim_steps=50,
)

betas = sample_fn("A tall person with a pear-shaped figure", n_samples=5)
# betas: numpy array (5, 10) in real SMPL units
```

`load_model_for_inference`:
1. Reads `text_dim` from the checkpoint
2. Reconstructs `TextCondBetaDiffusion` and loads denoiser + schedule weights
3. Strips any `_orig_mod.` prefix left by `torch.compile`
4. Lazy-loads Qwen text encoder on first call to `encode_text`

---

## 7. Hardware & Performance

Benchmarks on **RTX A5000** (24 GB VRAM, Ampere architecture).

### Training

| Phase | Time |
|---|---|
| Embedding cache (1M samples, one-time) | ~20–30 min |
| Per training epoch (denoiser only, BF16) | ~5–15 sec |
| 50 epochs total | ~5–12 min |
| **Total wall time** | **~30–45 min** |

### GPU memory during training

| Component | VRAM |
|---|---|
| Denoiser (~2M params, BF16) | ~4 MB |
| Cached embeddings (1M × 896, FP16) | ~1.8 GB |
| Batch (1024 × 896 text + 1024 × 10 betas) | ~3.7 MB |
| Gradients + optimiser states | ~50 MB |
| **Total** | **~2 GB** |

### Key optimisations

| Optimisation | Benefit |
|---|---|
| Pre-cached embeddings | ~5–10× fewer FLOPS per training step |
| TF32 matmul | ~2× matmul throughput on Ampere, free |
| BFloat16 autocast | ~1.5× throughput, no GradScaler needed |
| `torch.compile` (default mode) | Kernel fusion on denoiser MLP |
| Batch size 1024 | Better GPU utilisation |
| `T = 500` (vs 1000) | Halves per-epoch denoiser calls |
| Batched DDIM eval | Eliminates per-sample Python loop |
| Step-level LR warmup | Stable convergence with large batch |





---

## 8. Paper vs This Implementation

### 8.1 What the Paper Actually Does

The paper (Árbol & Casas, 2024) fine-tunes LLaMA-3 8B with QLoRA on 18,000 samples to produce beta values as text tokens. The LLM literally outputs a string like `"[1.131, 1.928, -2.347, ...]"` and three loss terms penalise it:

```
L = L_LLM          (token cross-entropy)
  + L_shape        (L1 on predicted vs true betas)
  + L_measurements (cross-entropy on SMPL-derived body measurements)
```

One description → one deterministic beta vector.

---

### 8.2 Difference-by-Difference Breakdown

| Dimension | Paper (Árbol & Casas) | This Implementation |
|---|---|---|
| Model type | Fine-tuned autoregressive LLM | Text-conditioned diffusion model (DDPM) |
| Base model | LLaMA-3 8B (trained with LoRA) | Qwen2.5-0.5B (fully frozen) |
| Trainable params | ~hundreds of millions (LoRA layers) | ~2M (denoiser MLP only) |
| Output per prompt | One deterministic beta vector | N diverse samples from a distribution |
| Handles ambiguity | Collapses to mean (MSE-like) | Models full P(β \| text) |
| Loss | Custom 3-term composite | Standard noise-prediction MSE |
| Dataset size | 18,000 training samples | 1,000,000 (56× larger) |
| Body shape types | Not present | Hourglass, Pear, Apple, Rectangle, Inverted Triangle |
| Weight vocabulary | Neutral language only | + colloquial terms (fat, thin, skinny, chubby…) |
| SMPL variant | SMPL-X (face + hands) | SMPL |
| Gender | Not supported (listed as future work) | Real-time toggle in GUI |
| CFG guidance | Not present | Adjustable at inference (1.0–8.0) |
| Rephrasing model | LLaMA-3 | Qwen2.5-1.5B (optional) |
| GUI | None | Full interactive PyVista 3D GUI |

---

### 8.3 Genuine Innovations

#### 1. Diffusion instead of LLM regression

The paper's core limitation is that it treats this as a sequence generation problem and outputs one answer. But *"tall athletic person"* is genuinely ambiguous — it describes a family of bodies, not one body.

This approach models it correctly:

```
Paper:  description → LLM → single β              (point estimate)
Ours:   description → encoder → DDPM → P(β|text)  (distribution)
```

The diffusion model can draw 5 or 10 plausible bodies from the same vague description, which the LLM approach simply cannot do.

#### 2. Classifier-Free Guidance

CFG lets you dial text fidelity vs diversity at inference time without retraining:

| Scale | Behaviour |
|---|---|
| `w = 1.0` | Unconditional sampling (ignores text, maximum diversity) |
| `w = 2.0` | Balanced (recommended default) |
| `w = 5.0+` | Strict text adherence, less body shape diversity |

The paper has no mechanism for this — their output is fixed once the model is trained.

#### 3. 56× larger dataset with richer vocabulary

The paper trains on 18K samples; this implementation uses 1M. The paper's descriptions use only neutral measurement language. This dataset adds:

- **5 canonical body shape types** (Hourglass, Pear, Apple, Rectangle, Inverted Triangle) derived from geometric ratios of shoulder/hip/waist measurements
- **Colloquial weight terms** (fat, chubby, skinny, slender, heavyset…) — how people actually describe bodies in natural language
- **`waist_width`** (lateral width at spine2) separate from `waist_depth`, enabling more geometrically accurate body shape classification

#### 4. Gender-conditioned generation

The paper explicitly lists gender support in Section 6 as future work:

> *"Another important avenue for future research is to train the network to recognize and appropriately respond to descriptions indicating the subject's gender… This involves teaching the network to understand the differences in SMPL-X shape parameters for different genders."*

This implementation addresses it directly: gender is selectable at inference time, SMPL is reloaded per gender, and the 3D mesh updates immediately.

#### 5. Computational efficiency

The paper fine-tunes 8B parameters with QLoRA on an RTX 4090. This approach:

- Keeps the text encoder frozen — only 2M params are trained
- Pre-caches all 1M embeddings once, so training only touches the MLP
- Full training in ~30–45 minutes on an A5000 vs the paper's multi-hour LLM fine-tuning run

---

### 8.4 Where the Paper Has an Advantage

**SMPL-X vs SMPL** — SMPL-X includes face expression and hand shape parameters, giving more complete avatars. This implementation uses SMPL (body only). This is a scope choice, not an oversight.

**Measurement-aware loss** — The paper's `L_measurements` term explicitly penalises incorrect BMI, height, and limb proportions in 3D space. The DDPM loss here is purely in parameter space. The paper's loss is a domain-specific constraint that helps especially on `bmi_very_low` and `shoulders_very_low` categories (per their ablation). Accuracy here comes from data scale and the diffusion framework instead.

---

### 8.5 Summary

| Claim | Defensible? |
|---|---|
| "We replicate the paper's approach" | No — the architecture is fundamentally different |
| "We extend the dataset with body shape types and colloquial vocabulary" | Yes, clearly novel |
| "We replace point-estimate regression with probabilistic diffusion" | Yes, genuine innovation |
| "We add CFG for inference-time diversity control" | Yes, no equivalent in the paper |
| "We address the paper's stated future work on gender" | Yes, directly |
| "We achieve this with 250× fewer trainable parameters" | Yes (~2M vs ~hundreds of millions) |

The paper fine-tunes a giant LLM to predict one answer. This system uses a small frozen encoder with a diffusion model to predict a distribution of answers — which is both more theoretically correct and more practically useful.