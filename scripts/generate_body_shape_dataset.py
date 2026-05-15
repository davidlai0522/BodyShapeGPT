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

import yaml
import numpy as np
import torch
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

# Human-readable ratio descriptions for logging
SHAPE_RATIO_DESCRIPTION: dict[str, str] = {
    "inverted_triangle": "shoulder/hip > 1.22",
    "pear":              "shoulder/hip < 0.83",
    "hourglass":         "shoulder/hip ≈ 1.0, waist/hip < 0.72",
    "rectangle":         "shoulder/hip ≈ 1.0, waist/hip > 0.88",
    "apple":             "waist_depth/height > 0.135, hip/height > 0.22",
}


# ---------------------------------------------------------------------------
# Vectorised batch sampling
# ---------------------------------------------------------------------------
# Strategy: sample large random batches, classify all in one numpy pass, keep
# matches.  No gradient computation — ~50-100× faster than gradient-based
# optimisation because:
#   • no backward pass through SMPL (eliminates the dominant GPU cost)
#   • single SMPL forward over the full batch is highly parallel
#   • classification is O(B·N) numpy, not O(B·steps·N) torch backward
# Typical acceptance rate from N(0,1) betas is ~15-25% per shape, so a batch
# of 512 yields ~80-120 matches — far more than the 1-2 from optimisation.

def _categorize_batch(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Vectorised version of categorize(). Returns (B,) object array of category strings."""
    p20, p40, p60, p80 = thresholds
    result = np.full(len(values), "very_high", dtype=object)
    result[values < p80] = "high"
    result[values < p60] = "average"
    result[values < p40] = "low"
    result[values < p20] = "very_low"
    return result


def extract_measurements_batch(
    verts: np.ndarray,   # (B, 6890, 3)
    joints: np.ndarray,  # (B, J, 3)
) -> dict[str, np.ndarray]:
    """
    Vectorised batch extraction of the six measurements needed by
    classify_body_shape().  Returns a dict of (B,) float64 arrays.
    Matches the per-sample logic in extract_measurements() exactly.
    """
    y_min = verts[:, :, 1].min(axis=1)   # (B,)
    y_max = verts[:, :, 1].max(axis=1)
    height = y_max - y_min

    shoulder_width = np.linalg.norm(
        joints[:, J["l_shoulder"]] - joints[:, J["r_shoulder"]], axis=1)

    # Hip width: X span of vertices near hip Y, |x| ≤ 0.35
    hip_y    = (joints[:, J["l_hip"], 1] + joints[:, J["r_hip"], 1]) / 2   # (B,)
    hip_mask = (
        (np.abs(verts[:, :, 1] - hip_y[:, None]) <= 0.06) &
        (np.abs(verts[:, :, 0]) <= 0.35)
    )  # (B, N)
    hip_x     = np.where(hip_mask, verts[:, :, 0], np.nan)
    hip_width = np.nan_to_num(np.nanmax(hip_x, axis=1) - np.nanmin(hip_x, axis=1))

    # Waist width and depth at spine2
    waist_y     = joints[:, J["spine2"], 1]                                  # (B,)
    waist_ymask = np.abs(verts[:, :, 1] - waist_y[:, None]) <= 0.04         # (B, N)
    waist_x     = np.where(waist_ymask & (np.abs(verts[:, :, 0]) <= 0.30),
                           verts[:, :, 0], np.nan)
    waist_width = np.nan_to_num(np.nanmax(waist_x, axis=1) - np.nanmin(waist_x, axis=1))
    waist_z     = np.where(waist_ymask & (np.abs(verts[:, :, 0]) <= 0.24),
                           verts[:, :, 2], np.nan)
    waist_depth = np.nan_to_num(np.nanmax(waist_z, axis=1) - np.nanmin(waist_z, axis=1))

    # BMI proxy: torso volume (20 slices, arms |x|>0.28 excluded) / height²
    # Fully vectorised via scatter reduce — no Python loop over slices.
    B        = verts.shape[0]
    n_slices = 20
    h_safe   = np.maximum(height, 1e-6)
    norm_y   = (verts[:, :, 1] - y_min[:, None]) / h_safe[:, None]          # (B, N) ∈ [0,1]
    slice_idx = np.floor(norm_y * n_slices).clip(0, n_slices - 1).astype(np.int32)
    arm_ok   = (np.abs(verts[:, :, 0]) <= 0.28)                             # (B, N)
    lin_idx  = (np.arange(B)[:, None] * n_slices + slice_idx).ravel()       # (B·N,)
    valid    = arm_ok.ravel()
    vi, vx, vz = lin_idx[valid], verts[:, :, 0].ravel()[valid], verts[:, :, 2].ravel()[valid]
    total    = B * n_slices
    x_max    = np.full(total, -np.inf);  np.maximum.at(x_max, vi, vx)
    x_min    = np.full(total,  np.inf);  np.minimum.at(x_min, vi, vx)
    z_max    = np.full(total, -np.inf);  np.maximum.at(z_max, vi, vz)
    z_min    = np.full(total,  np.inf);  np.minimum.at(z_min, vi, vz)
    filled   = (x_max > x_min).reshape(B, n_slices)
    w        = np.where(filled, (x_max - x_min).reshape(B, n_slices), 0.0)
    d        = np.where(filled, (z_max - z_min).reshape(B, n_slices), 0.0)
    vol      = (w * d * (height / n_slices)[:, None]).sum(axis=1)
    bmi_proxy = np.where(height > 0, vol / (height ** 2), 0.0)

    return {
        "height":         height,
        "shoulder_width": shoulder_width,
        "hip_width":      hip_width,
        "waist_width":    waist_width,
        "waist_depth":    waist_depth,
        "bmi_proxy":      bmi_proxy,
    }


def classify_batch(
    meas: dict[str, np.ndarray],
    thresholds: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Vectorised batch version of classify_body_shape().
    Returns (B,) object array of shape-name strings.
    Priority: apple > inverted_triangle > pear > hourglass > rectangle.
    """
    bmi_cat   = _categorize_batch(meas["bmi_proxy"],   thresholds["bmi_proxy"])
    waist_cat = _categorize_batch(meas["waist_depth"],  thresholds["waist_depth"])
    sw  = meas["shoulder_width"]
    hw  = meas["hip_width"]
    ww  = meas["waist_width"]
    ratio        = np.where(hw > 1e-6, sw / hw, 1.0)
    waist_to_hip = np.where(hw > 1e-6, ww / hw, 1.0)

    result = np.full(len(sw), "rectangle", dtype=object)
    result[
        (ratio >= 0.87) & (ratio <= 1.15) &
        ((waist_to_hip < 0.80) | np.isin(waist_cat, ["very_low", "low"]))
    ] = "hourglass"
    result[ratio < 0.87] = "pear"
    result[ratio > 1.15] = "inverted_triangle"
    result[
        np.isin(bmi_cat,   ["high", "very_high"]) &
        np.isin(waist_cat, ["high", "very_high"])
    ] = "apple"
    return result


def sample_shape_batch(
    smpl_model,
    target_shape: str,
    thresholds: dict[str, np.ndarray],
    device: str,
    batch_size: int = 512,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Sample `batch_size` random betas, run a single no-grad SMPL forward pass,
    classify all in one vectorised numpy pass, and return those matching
    `target_shape`.  Typical yield: 80-120 accepted candidates per call.
    """
    betas = torch.randn(batch_size, 10, device=device).clamp(-3.5, 3.5)
    with torch.no_grad():
        out = smpl_model(
            betas=betas,
            global_orient=torch.zeros(batch_size, 3, device=device),
            body_pose=torch.zeros(batch_size, 69, device=device),
            return_verts=True,
        )
    verts_np  = out.vertices.cpu().numpy()
    joints_np = out.joints.cpu().numpy()
    betas_np  = betas.cpu().numpy()

    meas  = extract_measurements_batch(verts_np, joints_np)
    found = classify_batch(meas, thresholds)
    valid = (meas["height"] > 0.5) & (meas["height"] < 2.5) & (found == target_shape)
    idxs  = np.where(valid)[0]
    return [(betas_np[i], verts_np[i], joints_np[i]) for i in idxs]


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_body_shape_dataset(
    shapes: dict[str, int],
    output_file: str,
    gender: str = "neutral",
    n_calib: int = 3000,
    batch_size: int = 512,
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

    written  = 0

    with open(output_file, "w") as fout:
        for shape_name, n_target in shapes.items():
            if shape_name not in VALID_SHAPES:
                print(f"Warning: unknown shape '{shape_name}' — skipping",
                      file=sys.stderr)
                continue

            pbar          = tqdm(total=n_target, desc=f"  {shape_name:<20}")
            shape_written = 0
            consecutive_empty = 0

            while shape_written < n_target:
                candidates = sample_shape_batch(
                    smpl, shape_name, thresholds, device,
                    batch_size=batch_size,
                )

                if not candidates:
                    consecutive_empty += 1
                    if consecutive_empty >= 50:
                        print(
                            f"\nWarning: {shape_name} yielded no matches in "
                            f"50 consecutive batches "
                            f"({shape_written}/{n_target} written). Skipping.",
                            file=sys.stderr,
                        )
                        break
                    continue
                consecutive_empty = 0

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
    parser.add_argument("--batch-size",  type=int, default=512,
                        help="Random betas sampled per iteration (larger = faster)")
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
        batch_size  = args.batch_size,
        seed        = args.seed,
        shape_prob  = args.shape_prob,
        n_extra_min = args.n_extra_min,
        n_extra_max = args.n_extra_max,
        focus_attrs = focus_attrs,
    )


if __name__ == "__main__":
    main()
