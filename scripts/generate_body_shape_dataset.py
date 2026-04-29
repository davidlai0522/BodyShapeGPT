"""
generate_body_shape_dataset.py — Body-Shape-Focused Dataset Generator

Unlike the main generator (which samples betas randomly and accepts whatever
shape falls out), this script uses **gradient-based optimisation** to find
SMPL betas that are geometrically guaranteed to produce a specific body shape
type.

How it works
------------
1. For each target shape a *differentiable loss* is defined in terms of soft,
   PyTorch-native body-proportion ratios:
     - shoulder_width / hip_width   (inverted-triangle vs pear)
     - waist_x_span / hip_width     (hourglass vs rectangle)
     - waist_depth / height         (apple: wide, heavy midsection)
2. A batch of random betas is optimised with Adam to minimise that loss plus
   an L2 regularisation term (keeps betas inside the training distribution).
3. Every candidate is *verified* with the reference classify_body_shape()
   from generate_dataset.py before being accepted.  Unverified candidates are
   discarded and the loop retries automatically.
4. Descriptions are generated with shape_prob=1.0 by default so every
   sentence explicitly names the body-shape type.

The output JSONL has the same schema as the main generator, with an extra
field  "body_shape": "<shape>"  for convenience.

Supported shapes
----------------
  hourglass, pear, apple, rectangle, inverted_triangle

Usage
-----
  # Equal split across all shapes, 5000 samples each:
  python generate_body_shape_dataset.py --shape all --n-per-shape 5000

  # Single shape:
  python generate_body_shape_dataset.py --shape hourglass --n 10000

  # Via config file:
  python generate_body_shape_dataset.py --config config/body_shape_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import smplx
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Re-use utilities from generate_dataset (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dataset import (  # noqa: E402
    J,
    MEAS_KEYS,
    CATEGORIES,
    ADJ,
    SHAPE_PHRASES,
    _apply_vocab_overrides,
    _load_config,
    _generate_description,
    calibrate_thresholds,
    measurements_to_categories,
    classify_body_shape,
    extract_measurements,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR   = _SCRIPT_DIR.parent

VALID_SHAPES: tuple[str, ...] = (
    "hourglass", "pear", "apple", "rectangle", "inverted_triangle",
)

# ---------------------------------------------------------------------------
# Differentiable body measurements
# ---------------------------------------------------------------------------
# All functions operate on batched tensors: verts (B, 6890, 3), joints (B, J, 3)
# and return (B,) scalar tensors so gradients flow back to betas.

_BETA_SOFT = 150.0   # logsumexp temperature for soft max/min


def _soft_span(coords: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation to the weighted span of `coords`.

    As β → ∞ this approaches  max_i(xᵢ | wᵢ > 0) − min_i(xᵢ | wᵢ > 0).
    Gaussian proximity weights naturally exclude distant vertices without
    non-differentiable boolean masking.

    coords, weights: (B, N)
    Returns: (B,)
    """
    log_w    = torch.log(weights.clamp_min(1e-8))
    soft_max = (1.0 / _BETA_SOFT) * torch.logsumexp(
        _BETA_SOFT * coords + log_w, dim=1)
    soft_min = -(1.0 / _BETA_SOFT) * torch.logsumexp(
        -_BETA_SOFT * coords + log_w, dim=1)
    return soft_max - soft_min


def diff_measurements(
    verts: torch.Tensor,
    joints: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute the key body-proportion measurements differentiably.

    verts:  (B, 6890, 3)
    joints: (B, J, 3)
    Returns a dict of (B,) tensors, all in metres.

    Measurements computed:
      height          — full-body Y extent
      shoulder_width  — joint-to-joint biacromial distance
      hip_width       — soft X span near hip Y level (excludes arms)
      waist_x_span    — soft lateral (X) span at spine2 level
      waist_depth     — soft front-to-back (Z) span at spine2 (excludes arms)
    """
    # ── Height (fully differentiable via autograd on max/min) ──────────────
    height = (verts[:, :, 1].max(dim=1).values
              - verts[:, :, 1].min(dim=1).values)

    # ── Shoulder width: joint-to-joint ─────────────────────────────────────
    shoulder_width = torch.norm(
        joints[:, J["l_shoulder"]] - joints[:, J["r_shoulder"]], dim=1)

    # ── Hip width: soft X span of vertices near hip Y, arm-free ───────────
    hip_y  = (joints[:, J["l_hip"], 1] + joints[:, J["r_hip"], 1]) / 2  # (B,)
    # Gaussian proximity weight along Y axis
    y_w_hip = torch.exp(-0.5 * ((verts[:, :, 1] - hip_y.unsqueeze(1)) / 0.06) ** 2)
    # Soft exclusion of arm vertices (large |X|)
    x_mask_hip = torch.exp(-((verts[:, :, 0].abs() / 0.35) ** 8))
    hip_width = _soft_span(verts[:, :, 0], y_w_hip * x_mask_hip)

    # ── Waist Y reference ──────────────────────────────────────────────────
    waist_y   = joints[:, J["spine2"], 1]  # (B,)
    y_w_waist = torch.exp(-0.5 * ((verts[:, :, 1] - waist_y.unsqueeze(1)) / 0.04) ** 2)

    # ── Waist lateral span: soft X span at spine2 ─────────────────────────
    x_mask_waist = torch.exp(-((verts[:, :, 0].abs() / 0.30) ** 8))
    waist_x_span = _soft_span(verts[:, :, 0], y_w_waist * x_mask_waist)

    # ── Waist depth: soft Z span at spine2, excluding arms ────────────────
    arm_mask    = torch.exp(-((verts[:, :, 0].abs() / 0.24) ** 8))
    waist_depth = _soft_span(verts[:, :, 2], y_w_waist * arm_mask)

    return {
        "height":         height,
        "shoulder_width": shoulder_width,
        "hip_width":      hip_width,
        "waist_x_span":   waist_x_span,
        "waist_depth":    waist_depth,
    }


# ---------------------------------------------------------------------------
# Per-shape loss functions
# ---------------------------------------------------------------------------
# Each function accepts the dict returned by diff_measurements() and returns
# a (B,) tensor of per-sample losses.  Lower = better match for the shape.
#
# Design rationale
# ----------------
# The classify_body_shape() thresholds are:
#   inverted_triangle  sw/hw > 1.15
#   pear               sw/hw < 0.87
#   apple              bmi_proxy HIGH + waist_depth HIGH
#   hourglass          0.87 ≤ sw/hw ≤ 1.15  AND  waist/hip < 0.80
#   rectangle          0.87 ≤ sw/hw ≤ 1.15  AND  waist/hip ≥ 0.80
#
# We push targets a little past the boundary so the verified shape is robust.

def _loss_inverted_triangle(m: dict[str, torch.Tensor]) -> torch.Tensor:
    """Penalise if shoulder_width / hip_width < 1.22 (target: > 1.22)."""
    ratio = m["shoulder_width"] / (m["hip_width"] + 1e-6)
    return F.relu(1.22 - ratio) ** 2 * 20.0


def _loss_pear(m: dict[str, torch.Tensor]) -> torch.Tensor:
    """Penalise if shoulder_width / hip_width > 0.83 (target: < 0.83)."""
    ratio = m["shoulder_width"] / (m["hip_width"] + 1e-6)
    return F.relu(ratio - 0.83) ** 2 * 20.0


def _loss_hourglass(m: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Shoulder ≈ hip (ratio ~ 1.0), waist_x_span / hip_width < 0.72.
    Bounds ensure we stay clear of the pear/inverted-triangle regions.
    """
    ratio  = m["shoulder_width"] / (m["hip_width"] + 1e-6)
    ww_hw  = m["waist_x_span"]   / (m["hip_width"] + 1e-6)
    balance = (ratio - 1.0) ** 2 * 15.0
    waist   = F.relu(ww_hw - 0.72) ** 2 * 25.0
    bounds  = (F.relu(ratio - 1.10) ** 2 + F.relu(0.90 - ratio) ** 2) * 10.0
    return balance + waist + bounds


def _loss_rectangle(m: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Shoulder ≈ hip (ratio ~ 1.0), waist_x_span / hip_width > 0.88
    (minimal waist definition).
    """
    ratio  = m["shoulder_width"] / (m["hip_width"] + 1e-6)
    ww_hw  = m["waist_x_span"]   / (m["hip_width"] + 1e-6)
    balance = (ratio - 1.0) ** 2 * 15.0
    waist   = F.relu(0.88 - ww_hw) ** 2 * 25.0
    bounds  = (F.relu(ratio - 1.10) ** 2 + F.relu(0.90 - ratio) ** 2) * 10.0
    return balance + waist + bounds


def _loss_apple(m: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Large waist_depth (front-to-back) and hip_width relative to height,
    implying high BMI and wide midsection — the two conditions for 'apple'.
    """
    waist_h = m["waist_depth"] / (m["height"] + 1e-6)
    hip_h   = m["hip_width"]   / (m["height"] + 1e-6)
    # Target: waist_depth > 13.5% of height, hip_width > 22% of height
    return (F.relu(0.135 - waist_h) ** 2 * 80.0
            + F.relu(0.22  - hip_h ) ** 2 * 40.0)


SHAPE_LOSS_FNS: dict[str, Callable] = {
    "inverted_triangle": _loss_inverted_triangle,
    "pear":              _loss_pear,
    "hourglass":         _loss_hourglass,
    "rectangle":         _loss_rectangle,
    "apple":             _loss_apple,
}

# Human-readable ratio descriptions for logging
SHAPE_RATIO_DESCRIPTION: dict[str, str] = {
    "inverted_triangle": "shoulder/hip > 1.22",
    "pear":              "shoulder/hip < 0.83",
    "hourglass":         "shoulder/hip ≈ 1.0, waist/hip < 0.72",
    "rectangle":         "shoulder/hip ≈ 1.0, waist/hip > 0.88",
    "apple":             "waist_depth/height > 0.135, hip/height > 0.22",
}


# ---------------------------------------------------------------------------
# Gradient-based optimisation
# ---------------------------------------------------------------------------

def _smpl_forward_diff(smpl_model, betas: torch.Tensor) -> object:
    """SMPL forward pass that keeps the autograd graph (for optimisation)."""
    bs     = betas.shape[0]
    device = betas.device
    return smpl_model(
        betas=betas,
        global_orient=torch.zeros(bs, 3, device=device),
        body_pose=torch.zeros(bs, 69, device=device),
        return_verts=True,
    )


def optimise_for_shape(
    smpl_model,
    target_shape: str,
    thresholds: dict[str, np.ndarray],
    device: str,
    n_restarts: int = 8,
    n_steps: int = 150,
    lr: float = 0.05,
    reg_weight: float = 0.05,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Optimise a batch of `n_restarts` random betas toward `target_shape`.

    Returns a list of (betas_np, verts_np, joints_np) tuples for every
    candidate that passes the classify_body_shape() verification check.
    Empty list means no candidate converged — the caller should retry.
    """
    loss_fn = SHAPE_LOSS_FNS[target_shape]

    # Random initialisation — start from N(0, I), clamped to valid range
    betas_param = (torch.randn(n_restarts, 10, device=device)
                   .clamp(-3.5, 3.5)
                   .clone()
                   .requires_grad_(True))
    optimizer = torch.optim.Adam([betas_param], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        clamped = betas_param.clamp(-3.5, 3.5)
        out     = _smpl_forward_diff(smpl_model, clamped)
        dmeas   = diff_measurements(out.vertices, out.joints)
        # Per-sample loss: shape loss + L2 reg on betas
        per_sample = loss_fn(dmeas) + reg_weight * (betas_param ** 2).mean(dim=1)
        per_sample.sum().backward()
        optimizer.step()

    # ── Verify with the reference (numpy) classifier ───────────────────────
    accepted: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        clamped   = betas_param.detach().clamp(-3.5, 3.5)
        out       = smpl_model(
            betas=clamped,
            global_orient=torch.zeros(n_restarts, 3, device=device),
            body_pose=torch.zeros(n_restarts, 69, device=device),
            return_verts=True,
        )
        verts_np  = out.vertices.cpu().numpy()   # (B, 6890, 3)
        joints_np = out.joints.cpu().numpy()     # (B, J, 3)
        betas_np  = clamped.cpu().numpy()        # (B, 10)

    for i in range(n_restarts):
        meas = extract_measurements(verts_np[i], joints_np[i])
        # Reject degenerate meshes
        if not (0.5 < meas["height"] < 2.5):
            continue
        if classify_body_shape(meas, thresholds) == target_shape:
            accepted.append((betas_np[i], verts_np[i], joints_np[i]))

    return accepted


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_body_shape_dataset(
    shapes: dict[str, int],
    output_file: str,
    gender: str = "neutral",
    n_calib: int = 3000,
    n_restarts: int = 8,
    n_steps: int = 150,
    lr: float = 0.05,
    reg_weight: float = 0.05,
    seed: int = 42,
    # Description style
    shape_prob: float = 1.0,
    n_extra_min: int = 1,
    n_extra_max: int = 2,
    focus_attrs: list[str] | None = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    smpl = smplx.create(
        str(_ROOT_DIR), model_type="smpl", gender=gender, num_betas=10,
    ).to(device)
    smpl.eval()

    print("Calibrating thresholds…")
    thresholds = calibrate_thresholds(smpl, device, n_calib=n_calib, batch_size=128)

    total = sum(shapes.values())
    print(f"\nTarget distribution:")
    for shape_name, n_target in shapes.items():
        ratio_desc = SHAPE_RATIO_DESCRIPTION.get(shape_name, "")
        print(f"  {shape_name:<22} {n_target:>6} samples   ({ratio_desc})")
    print(f"  {'TOTAL':<22} {total:>6}")
    print(f"\nOutput → {output_file}\n")

    written   = 0
    n_failed  = 0   # optimisation attempts that produced no verified candidate

    with open(output_file, "w") as fout:
        for shape_name, n_target in shapes.items():
            if shape_name not in VALID_SHAPES:
                print(f"Warning: unknown shape '{shape_name}' — skipping",
                      file=sys.stderr)
                continue

            pbar          = tqdm(total=n_target, desc=f"  {shape_name:<20}")
            shape_written = 0

            while shape_written < n_target:
                candidates = optimise_for_shape(
                    smpl, shape_name, thresholds, device,
                    n_restarts=n_restarts,
                    n_steps=n_steps,
                    lr=lr,
                    reg_weight=reg_weight,
                )

                if not candidates:
                    n_failed += n_restarts
                    continue

                for betas_np, verts_np, joints_np in candidates:
                    if shape_written >= n_target:
                        break

                    meas = extract_measurements(verts_np, joints_np)
                    cats = measurements_to_categories(meas, thresholds)
                    desc = _generate_description(
                        cats, shape_name,
                        shape_prob=shape_prob,
                        n_extra_min=n_extra_min,
                        n_extra_max=n_extra_max,
                        focus_attrs=focus_attrs,
                    )
                    record = {
                        "description":  desc,
                        "shape_params": str([round(float(b), 3) for b in betas_np]),
                        "body_shape":   shape_name,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    shape_written += 1
                    written       += 1
                    pbar.update(1)

            pbar.close()

    print(f"\nDone.  Wrote {written} samples to {output_file}.")
    if n_failed:
        print(f"  {n_failed} optimisation attempts did not converge to the "
              f"target shape (retried automatically).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Pre-parse for --config and --vocab-file ───────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config",     default=None)
    pre.add_argument("--vocab-file", default=None)
    pre_args, _ = pre.parse_known_args()

    cfg: dict = _load_config(pre_args.config) if pre_args.config else {}

    # Apply vocabulary overrides early (before description generation)
    vocab_cfg = dict(cfg)
    if pre_args.vocab_file:
        vocab_cfg["vocab_file"] = pre_args.vocab_file
    _apply_vocab_overrides(vocab_cfg)

    # ── Full argument parser ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="BodyShapeGPT Body-Shape-Focused Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default=None,
                        help="Path to a YAML/JSON config file")
    parser.add_argument("--vocab-file",  default=None,
                        help="Path to a YAML/JSON vocab override file")
    parser.add_argument(
        "--shape",
        choices=list(VALID_SHAPES) + ["all"],
        default="all",
        help="Target shape, or 'all' for an equal split across all shapes",
    )
    parser.add_argument("--n",           type=int, default=5000,
                        help="Total samples when --shape is a single shape")
    parser.add_argument("--n-per-shape", type=int, default=None,
                        help="Samples per shape when --shape all")
    parser.add_argument("--output",      default="body_shape_dataset.jsonl")
    parser.add_argument("--gender",
                        choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--n-calib",     type=int, default=3000,
                        help="Samples for threshold calibration")
    parser.add_argument("--n-restarts",  type=int, default=8,
                        help="Parallel optimisation restarts per batch")
    parser.add_argument("--n-steps",     type=int, default=150,
                        help="Adam steps per optimisation batch")
    parser.add_argument("--lr",          type=float, default=0.05,
                        help="Adam learning rate")
    parser.add_argument("--reg-weight",  type=float, default=0.05,
                        help="L2 regularisation weight on betas")
    parser.add_argument("--seed",        type=int, default=42)
    # Description style
    parser.add_argument("--shape-prob",  type=float, default=1.0,
                        help="Probability of naming the body-shape type in each description")
    parser.add_argument("--n-extra-min", type=int, default=1,
                        help="Min extra attributes mentioned besides height")
    parser.add_argument("--n-extra-max", type=int, default=2,
                        help="Max extra attributes mentioned besides height")
    parser.add_argument("--focus-attrs", type=str, default=None,
                        help="Comma-separated attributes always included in descriptions")

    if cfg:
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    # ── Build shapes dict ─────────────────────────────────────────────────
    # Priority: config `shapes:` key > --shape all > --shape <name>
    shapes_cfg = getattr(args, "shapes", None)
    if shapes_cfg and isinstance(shapes_cfg, dict):
        shapes = {k: int(v) for k, v in shapes_cfg.items()}
    elif args.shape == "all":
        n_each = args.n_per_shape or (args.n // len(VALID_SHAPES))
        shapes = {s: n_each for s in VALID_SHAPES}
    else:
        shapes = {args.shape: args.n}

    # ── focus_attrs: accept comma string (CLI) or list (config) ──────────
    raw_focus = getattr(args, "focus_attrs", None)
    if isinstance(raw_focus, str):
        focus_attrs = [a.strip() for a in raw_focus.split(",") if a.strip()] or None
    elif isinstance(raw_focus, list):
        focus_attrs = raw_focus or None
    else:
        focus_attrs = None

    generate_body_shape_dataset(
        shapes      = shapes,
        output_file = args.output,
        gender      = args.gender,
        n_calib     = args.n_calib,
        n_restarts  = args.n_restarts,
        n_steps     = args.n_steps,
        lr          = args.lr,
        reg_weight  = args.reg_weight,
        seed        = args.seed,
        shape_prob  = args.shape_prob,
        n_extra_min = args.n_extra_min,
        n_extra_max = args.n_extra_max,
        focus_attrs = focus_attrs,
    )


if __name__ == "__main__":
    main()
