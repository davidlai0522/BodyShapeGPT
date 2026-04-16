# Paper vs Your Implementation — What's the Same, What's Different, What's New

## What the Paper Actually Does

The paper fine-tunes LLaMA-3 8B with QLoRA on 18,000 samples to produce beta values as text tokens. The LLM literally outputs a string like `[1.131, 1.928, -2.347, ...]` and three loss terms penalise it:

* **L_LLM** (token cross-entropy)
* **L_shape** (L1 on predicted vs true betas)
* **L_measurements** (cross-entropy on SMPL-derived body measurements)

One description → one deterministic beta vector.

---

## Difference-by-Difference Breakdown

| Dimension | Paper (Árbol & Casas) | Your Implementation |
|---|---|---|
| **Model type** | Fine-tuned autoregressive LLM | Text-conditioned diffusion model (DDPM) |
| **Base model** | LLaMA-3 8B (trained with LoRA) | Qwen2.5 0.5B (fully frozen) |
| **Trainable params** | ~hundreds of millions (LoRA layers) | ~2M (denoiser MLP only) |
| **Output per prompt**| One deterministic beta vector | N diverse samples from a distribution |
| **Handles ambiguity**| Collapses to mean (MSE-like) | Models full P(β \| text) |
| **Loss** | Custom 3-term composite | Standard noise-prediction MSE |
| **Dataset size** | 18,000 training samples | 1,000,000 (56× larger) |
| **Body shape types** | Not present | Hourglass, Pear, Apple, Rectangle, Inverted Triangle |
| **Weight vocabulary**| Neutral language only | + colloquial terms (fat, thin, skinny, chubby…) |
| **SMPL variant** | SMPL-X (face + hands) | SMPL |
| **Gender** | Not supported (listed as future work) | Real-time toggle in GUI |
| **CFG guidance** | Not present | Adjustable at inference (1.0–8.0) |
| **Rephrasing model** | Llama-3 | Qwen2.5-1.5B (optional) |
| **GUI** | None | Full interactive PyVista 3D GUI |

---

## Genuine Innovations



### 1. Diffusion instead of LLM regression — the most fundamental change
The paper's core limitation is that it treats this as a sequence generation problem and outputs one answer. But "tall athletic person" is genuinely ambiguous — it describes a family of bodies, not one body.

Your approach models this correctly:
* **Paper:** description → LLM → single β (point estimate)
* **Yours:** description → encoder → DDPM → P(β\|text) (distribution)

This is not a minor tweak — it's a different problem formulation that is more theoretically sound. The diffusion model can draw 5 or 10 plausible bodies from the same vague description, which the LLM approach simply cannot do.

### 2. Classifier-Free Guidance (CFG) — a control knob the paper lacks


CFG lets you dial text fidelity vs diversity at inference time without retraining:
* **w = 1.0** → unconditional sampling (ignores text)
* **w = 2.0** → balanced (default)
* **w = 5.0+** → very literal text adherence

The paper has no mechanism for this. Their output is fixed once the model is trained.

### 3. 56× larger dataset with richer vocabulary
The paper trains on 18K samples. Yours uses 1M. More importantly, the paper's dataset descriptions use only neutral measurement language. Your dataset adds:
* 5 canonical body shape types (Hourglass, Pear, Apple, Rectangle, Inverted Triangle) derived from geometric ratios of shoulder/hip/waist measurements — the paper has no equivalent.
* Colloquial weight terms (fat, chubby, skinny, slender, heavyset…) which are how people actually describe bodies in natural language.
* A `waist_width` measurement (lateral width at spine2) separate from `waist_depth`, enabling more geometrically accurate body shape classification.

### 4. Gender-conditioned generation
This directly addresses the paper's listed future work. The paper explicitly lists this in Section 6:
> *"Another important avenue for future research is to train the network to recognize and appropriately respond to descriptions indicating the subject's gender... This involves teaching the network to understand the differences in SMPL-X shape parameters for different genders."*

Your implementation handles this: gender is selectable at inference time, SMPL is reloaded per gender, and the 3D mesh updates immediately.

### 5. Computational efficiency
The paper fine-tunes 8B parameters with QLoRA on an RTX 4090. Your approach:
* Keeps the text encoder frozen — only 2M params are trained.
* Pre-caches all 1M embeddings once, so training only touches the MLP.
* **Result:** full training in ~30–45 minutes on an A5000 vs the paper's multi-hour LLM fine-tuning run.

This isn't just an engineering convenience — it means the system is far more accessible and iterable.

---

## Where the Paper Has Something You Don't

* **SMPL-X vs SMPL:** SMPL-X includes face expression and hand shape parameters, giving more complete avatars. Your implementation uses SMPL (body only). This is a scope choice, not an oversight, but worth acknowledging.
* **Measurement-aware loss:** The paper's `L_measurements` term explicitly penalises incorrect BMI, height, and limb proportions in 3D space. Your DDPM loss is purely in parameter space. The paper's loss is a clever domain-specific constraint that may improve accuracy on rare extreme body shapes (their ablation shows it helps especially on `bmi_very_low` and `shoulders_very_low` categories). You don't have an equivalent — your accuracy comes from data scale and the diffusion framework instead.

---

## Summary for Your Presentation

| Claim | Defensible? |
|---|---|
| "We replicate the paper's approach" | No — the architecture is fundamentally different |
| "We extend the dataset with body shape types and colloquial vocabulary" | Yes, clearly novel |
| "We replace point-estimate regression with probabilistic diffusion" | Yes, genuine innovation |
| "We add CFG for inference-time diversity control" | Yes, no equivalent in the paper |
| "We address the paper's stated future work on gender" | Yes, directly |
| "We achieve this with 250× fewer trainable parameters" | Yes (~2M vs ~hundreds of millions) |