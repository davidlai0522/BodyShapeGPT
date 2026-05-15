"""
verify_body_shapes.py — Visual verification of body_shape_dataset.jsonl

For each shape type, samples N entries from the dataset, renders front and
side silhouettes of the SMPL mesh, and overlays the key body measurements
and the stored description.  Saves one PNG figure per shape.

Usage
-----
  python scripts/verify_body_shapes.py
  python scripts/verify_body_shapes.py --dataset body_shape_dataset.jsonl --n 3 --output verify_out/
  python scripts/verify_body_shapes.py --shape hourglass --n 5
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — works over SSH without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import torch
import smplx

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR   = _SCRIPT_DIR.parent

sys.path.insert(0, str(_SCRIPT_DIR))
from generate_dataset import J, extract_measurements, classify_body_shape, calibrate_thresholds

# ── Colour per shape ──────────────────────────────────────────────────────
SHAPE_COLORS = {
    "hourglass":         "#E07B6A",
    "pear":              "#6AADE0",
    "apple":             "#6AE09A",
    "rectangle":         "#C06AE0",
    "inverted_triangle": "#E0C46A",
}


# ---------------------------------------------------------------------------
# Mesh rendering helpers
# ---------------------------------------------------------------------------

def _project_silhouette(verts: np.ndarray, faces: np.ndarray,
                         axis_h: int, axis_v: int) -> list[np.ndarray]:
    """
    Return a list of triangle polygons projected onto (axis_h, axis_v).
    Used to draw front (axis_h=0 X, axis_v=1 Y) or side (axis_h=2 Z, axis_v=1 Y).
    """
    return [verts[f][:, [axis_h, axis_v]] for f in faces]


def _render_view(ax, verts: np.ndarray, faces: np.ndarray,
                 axis_h: int, axis_v: int,
                 title: str, color: str,
                 x_clip: float | None = None) -> None:
    """Draw a filled-mesh silhouette on ax.  x_clip: max |axis_h| to show."""
    tris   = _project_silhouette(verts, faces, axis_h, axis_v)
    patches = [Polygon(t, closed=True) for t in tris]
    col = PatchCollection(patches, facecolor=color, edgecolor=color, linewidth=0, alpha=0.85)
    ax.add_collection(col)
    h_vals = verts[:, axis_h]
    v_vals = verts[:, axis_v]
    if x_clip is not None:
        h_lo, h_hi = -x_clip, x_clip
    else:
        pad_h = (h_vals.max() - h_vals.min()) * 0.15
        h_lo  = h_vals.min() - pad_h
        h_hi  = h_vals.max() + pad_h
    pad_v  = (v_vals.max() - v_vals.min()) * 0.05
    ax.set_xlim(h_lo, h_hi)
    ax.set_ylim(v_vals.min() - pad_v, v_vals.max() + pad_v)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


def _meas_text(meas: dict[str, float], classified: str, expected: str) -> str:
    """Format the key measurements as a multiline string."""
    match = "✓" if classified == expected else f"✗ (got {classified})"
    lines = [
        f"shape: {expected}  {match}",
        f"height:         {meas['height']*100:.1f} cm",
        f"shoulder_width: {meas['shoulder_width']*100:.1f} cm",
        f"hip_width:      {meas['hip_width']*100:.1f} cm",
        f"waist_width:    {meas.get('waist_width', float('nan'))*100:.1f} cm",
        f"waist_depth:    {meas['waist_depth']*100:.1f} cm",
        f"bmi_proxy:      {meas['bmi_proxy']:.4f}",
    ]
    if meas["hip_width"] > 1e-6:
        sw_hw = meas["shoulder_width"] / meas["hip_width"]
        ww_hw = meas.get("waist_width", 0) / meas["hip_width"]
        lines += [
            f"sw/hw ratio:    {sw_hw:.3f}",
            f"ww/hw ratio:    {ww_hw:.3f}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_by_shape(dataset_path: str) -> dict[str, list[dict]]:
    """Load the JSONL and group records by body_shape."""
    by_shape: dict[str, list[dict]] = {}
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            shape = rec.get("body_shape", "unknown")
            by_shape.setdefault(shape, []).append(rec)
    return by_shape


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------

def verify(
    dataset_path: str,
    shapes: list[str],
    n_samples: int,
    output_dir: str,
    gender: str,
    seed: int,
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

    print("Calibrating thresholds for classification…")
    thresholds = calibrate_thresholds(smpl, device, n_calib=1000, batch_size=128)

    by_shape = load_by_shape(dataset_path)
    total_records = sum(len(v) for v in by_shape.values())
    print(f"\nDataset: {dataset_path}  ({total_records} records)")
    print("Shape counts: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_shape.items())))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mismatches_all = 0
    checked_all    = 0

    for shape in shapes:
        records = by_shape.get(shape, [])
        if not records:
            print(f"\n  [{shape}] — no records found in dataset, skipping")
            continue

        sample = random.sample(records, min(n_samples, len(records)))
        n      = len(sample)
        color  = SHAPE_COLORS.get(shape, "#888888")

        shape_dir = Path(output_dir) / shape
        shape_dir.mkdir(parents=True, exist_ok=True)

        mismatches = 0
        for idx, rec in enumerate(sample):
            fig, axes = plt.subplots(
                1, 3,
                figsize=(12, 5),
                gridspec_kw={"width_ratios": [1, 1, 1.4]},
            )
            ax_front, ax_side, ax_text = axes

            fig.suptitle(
                f"{shape}  —  sample {idx + 1}/{n}",
                fontsize=12, fontweight="bold", color=color,
            )

            betas_list = ast.literal_eval(rec["shape_params"])
            betas = torch.tensor([betas_list], dtype=torch.float32, device=device)

            with torch.no_grad():
                out = smpl(
                    betas=betas,
                    global_orient=torch.zeros(1, 3, device=device),
                    body_pose=torch.zeros(1, 69, device=device),
                    return_verts=True,
                )
            verts  = out.vertices[0].cpu().numpy()
            joints = out.joints[0].cpu().numpy()
            faces  = smpl.faces

            meas       = extract_measurements(verts, joints)
            classified = classify_body_shape(meas, thresholds)
            if classified != shape:
                mismatches += 1

            _render_view(ax_front, verts, faces, axis_h=0, axis_v=1,
                         title="Front", color=color, x_clip=0.30)
            _render_view(ax_side,  verts, faces, axis_h=2, axis_v=1,
                         title="Side",  color=color)

            meas_str = _meas_text(meas, classified, shape)
            desc_wrapped = "\n".join(
                rec["description"][i:i+42] for i in range(0, len(rec["description"]), 42)
            )
            ax_text.axis("off")
            ax_text.text(
                0.05, 0.97, meas_str + "\n\n" + desc_wrapped,
                transform=ax_text.transAxes,
                fontsize=8, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8),
            )

            fig.tight_layout()
            out_path = shape_dir / f"sample_{idx + 1:03d}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)

        mismatch_note = (
            f"  ⚠ {mismatches}/{n} misclassified" if mismatches else f"  ✓ all {n} verified"
        )
        print(f"\n  [{shape}]{mismatch_note}  →  {shape_dir}/")

        mismatches_all += mismatches
        checked_all    += n

    print(f"\nSummary: {checked_all - mismatches_all}/{checked_all} passed classification check")
    print(f"Figures saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

VALID_SHAPES = ("hourglass", "pear", "apple", "rectangle", "inverted_triangle")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify body_shape_dataset integrity with visual SMPL renders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="body_shape_dataset.jsonl",
                        help="Path to the JSONL dataset file")
    parser.add_argument("--shape",
                        choices=list(VALID_SHAPES) + ["all"],
                        default="all",
                        help="Shape to verify, or 'all'")
    parser.add_argument("--n",       type=int, default=3,
                        help="Samples to visualise per shape")
    parser.add_argument("--output",  default="verify_out",
                        help="Directory to save PNG figures")
    parser.add_argument("--gender",
                        choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    shapes = list(VALID_SHAPES) if args.shape == "all" else [args.shape]

    verify(
        dataset_path = args.dataset,
        shapes       = shapes,
        n_samples    = args.n,
        output_dir   = args.output,
        gender       = args.gender,
        seed         = args.seed,
    )


if __name__ == "__main__":
    main()
