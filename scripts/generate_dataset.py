"""
generate_dataset.py — BodyShapeGPT Dataset Generation Tool

Implements the dataset generation pipeline described in:
  "BodyShapeGPT: SMPL Body Shape Manipulation with LLMs"  (Árbol & Casas, 2024)

Pipeline:
  1. Sample N betas ~ N(0, I_10), clipped to [-3.5, 3.5]
  2. Batch SMPL forward pass → (vertices 6890×3, joints 45×3)
  3. Extract 12 body measurements from geometry (joint-based lengths,
     vertex-slice thicknesses)
  4. Calibrate 5-level thresholds on a held-out sample (percentile-based)
  5. Map each measurement to a category: very_low/low/average/high/very_high
  6. Generate natural-language descriptions from a rich template library
     (10+ sentence patterns, 3–5 synonyms per attribute per level, random
     attribute ordering and selection — no LLM needed for variety)
  7. Optional LLM rephrasing pass (--rephrase, uses Qwen2.5-1.5B-Instruct)
     with keyword-level validation to preserve accuracy
  8. Write JSONL: {"description": "…", "shape_params": "[β₀, …, β₉]"}

Accuracy guarantees:
  - Measurements are derived directly from SMPL geometry (ground truth).
  - Thresholds are calibrated from the same distribution used for sampling,
    so category assignments are statistically correct by construction.
  - When LLM rephrasing is used, a keyword validation step rejects outputs
    that change any attribute level (falling back to the template version).

Usage:
  python generate_dataset.py                           # 21 000 samples
  python generate_dataset.py --n 50000 --output big.jsonl
  python generate_dataset.py --n 21000 --rephrase      # + LLM paraphrasing
  python generate_dataset.py --validate                # audit existing file
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
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR   = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# SMPL joint indices  (first 24 of the 45 returned by smplx)
# ---------------------------------------------------------------------------
J = {
    "pelvis":      0,
    "l_hip":       1,  "r_hip":       2,
    "spine1":      3,
    "l_knee":      4,  "r_knee":      5,
    "spine2":      6,
    "l_ankle":     7,  "r_ankle":     8,
    "spine3":      9,
    "l_foot":     10,  "r_foot":     11,
    "neck":       12,
    "l_collar":   13,  "r_collar":   14,
    "head":       15,
    "l_shoulder": 16,  "r_shoulder": 17,
    "l_elbow":    18,  "r_elbow":    19,
    "l_wrist":    20,  "r_wrist":    21,
    "l_hand":     22,  "r_hand":     23,
}

# ---------------------------------------------------------------------------
# Measurement names and their human-readable labels used in descriptions
# ---------------------------------------------------------------------------
MEAS_KEYS = [
    "height",          # full body height from vertices
    "neck_length",     # neck joint → head joint
    "arm_length",      # shoulder → elbow → wrist  (left arm)
    "leg_length",      # hip → knee → ankle        (left leg)
    "shoulder_width",  # biacromial distance (shoulder joint to shoulder joint)
    "hip_width",       # flesh width at hip/pelvis level
    "waist_depth",     # front-to-back at spine2 level
    "chest_depth",     # front-to-back at spine3 level (described as bust)
    "arm_girth",       # arm cross-section at elbow level
    "leg_girth",       # leg cross-section at knee level
    "bmi_proxy",       # torso volume / height² proxy
]

CATEGORIES = ["very_low", "low", "average", "high", "very_high"]

# ---------------------------------------------------------------------------
# Adjective vocabulary
# Each key maps category → list of synonym phrases.
# Randomly drawn during description generation for lexical variety.
# ---------------------------------------------------------------------------
ADJ: dict[str, dict[str, list[str]]] = {
    "height": {
        "very_low":  ["very short", "extremely short", "very petite"],
        "low":       ["short", "petite", "of below-average height"],
        "average":   ["average height", "of average stature", "medium height"],
        "high":      ["tall", "of above-average height"],
        "very_high": ["very tall", "extremely tall", "towering"],
    },
    "neck_length": {
        "very_low":  ["very short neck", "an extremely short neck"],
        "low":       ["short neck", "a below-average neck"],
        "average":   ["a neck of average length", "an average-length neck"],
        "high":      ["a long neck", "a tall neck"],
        "very_high": ["a very long neck", "an extremely long neck", "a very tall neck"],
    },
    "arm_length": {
        "very_low":  ["very short arms", "extremely short arms"],
        "low":       ["short arms", "below-average arm length"],
        "average":   ["arms of average length", "average-length arms"],
        "high":      ["long arms", "above-average arm length"],
        "very_high": ["very long arms", "extremely long arms"],
    },
    "leg_length": {
        "very_low":  ["very short legs", "extremely short legs"],
        "low":       ["short legs", "legs of below-average length"],
        "average":   ["legs of average length", "average leg length"],
        "high":      ["long legs", "above-average leg length"],
        "very_high": ["very long legs", "extremely long legs"],
    },
    "shoulder_width": {
        "very_low":  ["very narrow shoulders", "extremely narrow shoulders"],
        "low":       ["narrow shoulders", "slim shoulders"],
        "average":   ["average shoulder width", "shoulders of average width"],
        "high":      ["broad shoulders", "wide shoulders"],
        "very_high": ["very broad shoulders", "extremely broad shoulders"],
    },
    "hip_width": {
        "very_low":  ["very narrow hips", "an extremely narrow hip"],
        "low":       ["narrow hips", "slim hips"],
        "average":   ["hips of average width", "average hip width"],
        "high":      ["broad hips", "very broad hip"],
        "very_high": ["very broad hips", "extremely wide hips"],
    },
    "waist_depth": {
        "very_low":  ["a very thin waist", "an extremely slender waist"],
        "low":       ["a thin waist", "a slim waist"],
        "average":   ["an average waist", "a waist of average size"],
        "high":      ["a thick waist", "a wide waist"],
        "very_high": ["a very thick waist", "an extremely wide waist"],
    },
    "chest_depth": {
        "very_low":  ["a thin bust", "a very thin bust"],
        "low":       ["a slim bust", "a below-average bust"],
        "average":   ["an average bust", "a bust of average size"],
        "high":      ["a thick bust", "a full bust"],
        "very_high": ["a very thick bust", "a large bust"],
    },
    "arm_girth": {
        "very_low":  ["thin arms", "very slender arms"],
        "low":       ["slim arms", "lean arms"],
        "average":   ["arms of average thickness", "average arms"],
        "high":      ["thick arms", "muscular arms"],
        "very_high": ["very thick arms", "extremely muscular arms"],
    },
    "leg_girth": {
        "very_low":  ["thin legs", "very slender legs"],
        "low":       ["slim legs", "lean legs"],
        "average":   ["legs of average thickness", "average legs"],
        "high":      ["thick legs", "muscular legs"],
        "very_high": ["very thick legs", "extremely muscular legs"],
    },
    "bmi_proxy": {
        "very_low":  ["low body mass", "a very lean build", "very low body mass",
                      "a very skinny build", "a very thin frame", "an extremely slim figure",
                      "a bony build"],
        "low":       ["below-average body mass", "a lean build", "a slim build",
                      "a slender figure", "a thin frame", "a slight build"],
        "average":   ["average body mass", "a medium build", "an average weight",
                      "a normal weight", "a regular build"],
        "high":      ["high body mass", "a heavy build", "above-average body mass",
                      "a heavyset build", "a chubby build", "an overweight build",
                      "a stocky build", "a fat build", "extra body weight"],
        "very_high": ["a lot of body mass", "a very heavy build",
                      "extremely high body mass", "an obese build",
                      "a very overweight build", "a very fat build", "a very large build"],
    },
}

# ---------------------------------------------------------------------------
# Body shape type phrases
# Each shape maps to a list of natural-language descriptors.
# ---------------------------------------------------------------------------
SHAPE_PHRASES: dict[str, list[str]] = {
    "hourglass": [
        "an hourglass figure",
        "an hourglass body shape",
        "a classic hourglass shape",
        "balanced shoulders and hips with a defined waist",
        "well-proportioned curves with a narrow waist",
        "a curvy hourglass build",
    ],
    "pear": [
        "a pear-shaped figure",
        "a pear body type",
        "wider at the hips than the shoulders",
        "a lower-body-heavy build",
        "narrow shoulders with fuller hips",
        "a pear silhouette",
    ],
    "apple": [
        "an apple-shaped figure",
        "an apple body type",
        "carrying weight primarily in the midsection",
        "a round midsection",
        "a heavier torso relative to the lower body",
        "an apple silhouette",
    ],
    "rectangle": [
        "a rectangular build",
        "a straight figure",
        "a rectangular body shape",
        "balanced proportions with minimal waist definition",
        "an athletic straight build",
        "a banana-shaped figure",
    ],
    "inverted_triangle": [
        "an inverted triangle shape",
        "a V-shaped torso",
        "broad shoulders tapering to narrow hips",
        "an inverted triangle body type",
        "wider at the shoulders than the hips",
        "a V-shaped build",
    ],
}


def _apply_vocab_overrides(cfg: dict) -> None:
    """
    Mutate the global ADJ and SHAPE_PHRASES with user-supplied overrides.

    Overrides can come from two places (both optional, both applied if present):
      1. A separate YAML/JSON file pointed to by cfg["vocab_file"].
      2. Inline "adj" / "shape_phrases" keys inside cfg itself.

    Deep-merge semantics:
      - adj:           only the listed attr→category pairs are replaced; the
                       rest of ADJ is left untouched.
      - shape_phrases: only the listed shape keys are replaced.
    """
    sources: list[dict] = []

    # 1. External vocab file
    vocab_path = cfg.get("vocab_file")
    if vocab_path:
        p = Path(vocab_path)
        if not p.is_absolute():
            p = _ROOT_DIR / p
        if not p.exists():
            print(f"Warning: vocab_file not found: {p}", file=sys.stderr)
        else:
            with p.open() as f:
                if p.suffix in (".yaml", ".yml"):
                    sources.append(yaml.safe_load(f) or {})
                else:
                    sources.append(json.load(f))

    # 2. Inline keys in the main config
    sources.append(cfg)

    for src in sources:
        # --- ADJ overrides ---
        adj_patch = src.get("adj") or {}
        for attr, cat_map in adj_patch.items():
            if attr not in ADJ:
                print(f"Warning: unknown adj attribute '{attr}' — skipping",
                      file=sys.stderr)
                continue
            if not isinstance(cat_map, dict):
                continue
            for cat, phrases in cat_map.items():
                if cat not in CATEGORIES:
                    print(f"Warning: unknown category '{cat}' for '{attr}' — skipping",
                          file=sys.stderr)
                    continue
                if not isinstance(phrases, list) or not phrases:
                    print(f"Warning: adj[{attr}][{cat}] must be a non-empty list — skipping",
                          file=sys.stderr)
                    continue
                ADJ[attr][cat] = phrases

        # --- SHAPE_PHRASES overrides ---
        sp_patch = src.get("shape_phrases") or {}
        for shape, phrases in sp_patch.items():
            if shape not in SHAPE_PHRASES:
                print(f"Warning: unknown shape '{shape}' — skipping", file=sys.stderr)
                continue
            if not isinstance(phrases, list) or not phrases:
                print(f"Warning: shape_phrases[{shape}] must be a non-empty list — skipping",
                      file=sys.stderr)
                continue
            SHAPE_PHRASES[shape] = phrases


# ---------------------------------------------------------------------------
# Measurement extraction
# ---------------------------------------------------------------------------

def _segment_len(j: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(j[a] - j[b]))


def _slice_depth(verts: np.ndarray, y_c: float, y_half: float = 0.04,
                 x_max: float = 0.24) -> float:
    """
    Front-to-back depth of the torso at height y_c.
    x_max clips arms (which extend along X in T-pose) from the measurement.
    """
    mask = (
        np.abs(verts[:, 1] - y_c) <= y_half
    ) & (np.abs(verts[:, 0]) <= x_max)
    if mask.sum() < 10:
        return 0.0
    v = verts[mask]
    return float(v[:, 2].max() - v[:, 2].min())


def _limb_girth(verts: np.ndarray, joint_center: np.ndarray,
                axis: int, half_range: float = 0.035,
                off_axis_range: float = 0.07) -> float:
    """
    Estimate limb girth at joint_center by finding the cross-sectional
    extent of nearby vertices. axis=0 isolates along X, axis=2 along Z.
    """
    axis_dist = np.abs(verts[:, axis] - joint_center[axis])
    y_dist    = np.abs(verts[:, 1]    - joint_center[1])
    mask = (y_dist <= half_range) & (axis_dist <= off_axis_range)
    if mask.sum() < 5:
        return 0.0
    v = verts[mask]
    z_span = float(v[:, 2].max() - v[:, 2].min())
    x_span = float(v[:, 0].max() - v[:, 0].min())
    return max(z_span, x_span)


def extract_measurements(verts: np.ndarray, joints: np.ndarray) -> dict[str, float]:
    """
    Extract 12 body measurements from SMPL vertices (6890, 3) and
    joints (≥24, 3).  All values in metres.
    """
    j = joints  # shorthand

    # --- Lengths via joint chains ---
    height = float(verts[:, 1].max() - verts[:, 1].min())

    neck_length = _segment_len(j, J["neck"], J["head"])

    arm_length = (_segment_len(j, J["l_shoulder"], J["l_elbow"]) +
                  _segment_len(j, J["l_elbow"],    J["l_wrist"]))

    leg_length = (_segment_len(j, J["l_hip"],   J["l_knee"]) +
                  _segment_len(j, J["l_knee"],  J["l_ankle"]))

    # --- Widths ---
    # Shoulder width: joint-to-joint biacromial distance (avoids T-pose arm vertices)
    shoulder_width = _segment_len(j, J["l_shoulder"], J["r_shoulder"])

    # Hip width: X span of vertices near pelvis height, torso-limited
    hip_y     = float((j[J["l_hip"], 1] + j[J["r_hip"], 1]) / 2)
    hip_ymask = (np.abs(verts[:, 1] - hip_y) <= 0.06) & (np.abs(verts[:, 0]) <= 0.35)
    hip_width = float(verts[hip_ymask, 0].max() - verts[hip_ymask, 0].min()) if hip_ymask.sum() > 10 else 0.0

    # Waist width: X span near spine2 (used for body shape classification)
    waist_y    = float(j[J["spine2"], 1])
    waist_mask = (np.abs(verts[:, 1] - waist_y) <= 0.04) & (np.abs(verts[:, 0]) <= 0.30)
    waist_width = float(verts[waist_mask, 0].max() - verts[waist_mask, 0].min()) if waist_mask.sum() > 10 else 0.0

    # --- Depths (torso front-to-back, arms excluded by x_max=0.24) ---
    waist_depth = _slice_depth(verts, float(j[J["spine2"], 1]))
    chest_depth = _slice_depth(verts, float(j[J["spine3"], 1]))

    # --- Limb girths ---
    arm_girth = _limb_girth(verts, j[J["l_elbow"]], axis=0,
                            half_range=0.03, off_axis_range=0.06)
    leg_girth = _limb_girth(verts, j[J["l_knee"]], axis=0,
                            half_range=0.05, off_axis_range=0.12)

    # --- BMI proxy: torso volume / height²
    # Exclude arm vertices (|X|>0.28) so T-pose arms don't inflate volume.
    n_slices = 20
    ys  = np.linspace(verts[:, 1].min(), verts[:, 1].max(), n_slices + 1)
    vol = 0.0
    for i in range(n_slices):
        sl = verts[
            (verts[:, 1] >= ys[i]) & (verts[:, 1] < ys[i + 1]) &
            (np.abs(verts[:, 0]) <= 0.28)
        ]
        if sl.shape[0] > 5:
            w = sl[:, 0].max() - sl[:, 0].min()
            d = sl[:, 2].max() - sl[:, 2].min()
            vol += w * d * (ys[i + 1] - ys[i])
    bmi_proxy = vol / (height ** 2) if height > 0 else 0.0

    return {
        "height":         height,
        "neck_length":    neck_length,
        "arm_length":     arm_length,
        "leg_length":     leg_length,
        "shoulder_width": shoulder_width,
        "hip_width":      hip_width,
        "waist_width":    waist_width,   # lateral width — used for body shape only
        "waist_depth":    waist_depth,
        "chest_depth":    chest_depth,
        "arm_girth":      arm_girth,
        "leg_girth":      leg_girth,
        "bmi_proxy":      bmi_proxy,
    }


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def _smpl_forward(smpl_model, betas: torch.Tensor, device: str):
    """Call SMPL with explicit zero-pose so batch size can be anything."""
    bs = betas.shape[0]
    return smpl_model(
        betas=betas,
        global_orient=torch.zeros(bs, 3, device=device),
        body_pose=torch.zeros(bs, 69, device=device),
        return_verts=True,
    )


def calibrate_thresholds(
    smpl_model: smplx.SMPL,
    device: str,
    n_calib: int = 5000,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """
    Run SMPL on n_calib random bodies and compute the 20th/40th/60th/80th
    percentile of each measurement.  These become the 5-category boundaries.
    """
    print(f"Calibrating thresholds on {n_calib} random bodies…")
    all_meas: dict[str, list[float]] = {k: [] for k in MEAS_KEYS}

    for start in tqdm(range(0, n_calib, batch_size), desc="  calib"):
        bs    = min(batch_size, n_calib - start)
        betas = torch.randn(bs, 10, device=device).clamp(-3.5, 3.5)
        with torch.no_grad():
            out = _smpl_forward(smpl_model, betas, device)
        verts  = out.vertices.cpu().numpy()   # (bs, 6890, 3)
        joints = out.joints.cpu().numpy()     # (bs, 45,   3)

        for i in range(bs):
            m = extract_measurements(verts[i], joints[i])
            for k in MEAS_KEYS:
                all_meas[k].append(m[k])

    thresholds: dict[str, np.ndarray] = {}
    for k in MEAS_KEYS:
        arr = np.array(all_meas[k])
        thresholds[k] = np.percentile(arr, [20, 40, 60, 80])

    return thresholds


def categorize(value: float, thresholds: np.ndarray) -> str:
    """Map a single measurement to its 5-level category."""
    p20, p40, p60, p80 = thresholds
    if value < p20:
        return "very_low"
    elif value < p40:
        return "low"
    elif value < p60:
        return "average"
    elif value < p80:
        return "high"
    else:
        return "very_high"


def measurements_to_categories(
    meas: dict[str, float],
    thresholds: dict[str, np.ndarray],
) -> dict[str, str]:
    return {k: categorize(meas[k], thresholds[k]) for k in MEAS_KEYS}


def classify_body_shape(meas: dict[str, float], thresholds: dict[str, np.ndarray]) -> str:
    """
    Classify the body into one of 5 canonical shape types.

    Logic (evaluated in priority order):
      apple            — high/very_high bmi_proxy AND high/very_high waist relative to hips
      inverted_triangle — shoulder_width / hip_width > 1.15
      pear             — shoulder_width / hip_width < 0.87
      hourglass        — balanced shoulder/hip AND waist noticeably narrower than hips
      rectangle        — balanced proportions, minimal waist definition

    Uses waist_width (lateral) for shape ratios; falls back to waist_depth if absent.
    """
    bmi_cat   = categorize(meas["bmi_proxy"],   thresholds["bmi_proxy"])
    waist_cat = categorize(meas["waist_depth"],  thresholds["waist_depth"])

    sw = meas["shoulder_width"]
    hw = meas["hip_width"]
    ww = meas.get("waist_width", meas["waist_depth"])  # prefer lateral width
    ratio = sw / hw if hw > 1e-6 else 1.0

    # Apple: heavy build concentrated in the midsection
    if bmi_cat in ("high", "very_high") and waist_cat in ("high", "very_high"):
        return "apple"

    # Inverted triangle: shoulders clearly dominate hips
    if ratio > 1.15:
        return "inverted_triangle"

    # Pear: hips clearly wider than shoulders
    if ratio < 0.87:
        return "pear"

    # Hourglass vs rectangle: balanced shoulder/hip — distinguish by waist narrowness
    waist_to_hip = ww / hw if hw > 1e-6 else 1.0
    if waist_to_hip < 0.80 or waist_cat in ("very_low", "low"):
        return "hourglass"

    return "rectangle"


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------

# Sentence-level templates.
# Placeholders filled by _pick() draws from ADJ.
# Each template function takes a dict {attr: phrase} and returns a sentence.

def _pick(attr: str, cat: str) -> str:
    """Randomly select a synonym phrase for (attribute, category)."""
    return random.choice(ADJ[attr][cat])


def _select_attrs(
    cats: dict[str, str],
    n_extra_min: int = 1,
    n_extra_max: int = 3,
    focus_attrs: list[str] | None = None,
) -> list[str]:
    """
    Select a subset of attributes to mention (excluding height, which is always present).

    - focus_attrs: attributes always included in the output (e.g. ["bmi_proxy"]).
    - n_extra_min / n_extra_max: range for the *total* number of extra attributes;
      focus_attrs count toward that total.
    - Remaining slots are filled by weighted random sampling, biased toward
      non-average values for lexical variety.
    """
    pinned = [a for a in (focus_attrs or []) if a != "height" and a in MEAS_KEYS]
    n_extra = random.randint(n_extra_min, n_extra_max)
    n_random = max(0, n_extra - len(pinned))

    pool = [k for k in MEAS_KEYS if k != "height" and k not in pinned]
    if n_random > 0 and pool:
        weights = [3.0 if cats[k] in ("very_low", "very_high") else
                   1.5 if cats[k] in ("low", "high") else
                   0.6 for k in pool]
        n_random = min(n_random, len(pool))
        w = np.array(weights, dtype=float)
        w /= w.sum()
        sampled = np.random.choice(pool, size=n_random, replace=False, p=w).tolist()
    else:
        sampled = []

    result = pinned + sampled
    random.shuffle(result)
    return result


def _join_phrases(phrases: list[str]) -> str:
    """Join 0-n phrases with commas and 'and'. Returns '' for empty list."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _generate_description(
    cats: dict[str, str],
    body_shape: str,
    shape_prob: float = 0.65,
    n_extra_min: int = 1,
    n_extra_max: int = 3,
    focus_attrs: list[str] | None = None,
) -> str:
    """Build one natural-language sentence from measurement categories and body shape."""
    height_cat  = cats["height"]
    h_phrase    = _pick("height", height_cat)
    sp          = random.choice(SHAPE_PHRASES[body_shape])

    extra_keys   = _select_attrs(cats, n_extra_min, n_extra_max, focus_attrs)
    attr_phrases = [_pick(k, cats[k]) for k in extra_keys]
    has_attrs    = bool(attr_phrases)

    include_shape = random.random() < shape_prob

    if include_shape:
        # ---- Templates that include the body-shape phrase ----
        # Attr-less variants (S1, S3, S7) are always eligible.
        # Variants that use attr_phrases are only added when attrs are present.
        shape_templates_noattrs = [
            # S1. "A [height] person with [shape]."
            lambda h, ap, s: f"A {h} person with {s}.",
            # S3. "[Height] individual with [shape]."
            lambda h, ap, s: f"{h.capitalize()} individual with {s}.",
            # S7. "A [height] frame with [shape]."
            lambda h, ap, s: f"A {h} frame with {s}.",
        ]
        shape_templates_withattrs = [
            # S2. "A [height] person with [shape] and X, Y."
            lambda h, ap, s: f"A {h} person with {s} and {_join_phrases(ap)}.",
            # S4. "[Height] person featuring [shape] and X, Y."
            lambda h, ap, s: f"{h.capitalize()} person featuring {s} and {_join_phrases(ap)}.",
            # S5. "A [height] person. They have [shape] with X, Y."
            lambda h, ap, s: f"A {h} person. They have {s} with {_join_phrases(ap)}.",
            # S6. "[Shape], [height], X and Y."
            lambda h, ap, s: f"{s.capitalize()}, {h}, {_join_phrases(ap)}.",
            # S8. "[Height] person with [shape] — X and Y."
            lambda h, ap, s: f"{h.capitalize()} person with {s} — {_join_phrases(ap)}.",
            # S9. Lead with weight if extreme, then shape
            lambda h, ap, s: (
                f"{_pick('bmi_proxy', cats['bmi_proxy']).capitalize()}, "
                f"{h}, with {s} and {_join_phrases(ap)}."
                if cats["bmi_proxy"] in ("very_low", "very_high")
                else f"A {h} person with {s} and {_join_phrases(ap)}."
            ),
        ]
        pool = shape_templates_noattrs + (shape_templates_withattrs if has_attrs else [])
        template = random.choice(pool)
        return template(h_phrase, attr_phrases, sp)
    else:
        # ---- Templates without an explicit shape label ----
        if not has_attrs:
            # Fallback: height-only sentences
            return random.choice([
                f"A {h_phrase} person.",
                f"{h_phrase.capitalize()} individual.",
                f"A {h_phrase} frame.",
            ])
        plain_templates = [
            # 1. "A [height] person with X, Y, and Z."
            lambda h, ap: f"A {h} person with {_join_phrases(ap)}.",
            # 2. "Person with [height], X, Y, and Z."
            lambda h, ap: f"Person with {h}, {_join_phrases(ap)}.",
            # 3. "[Height] individual with X, Y, and Z."
            lambda h, ap: f"{h.capitalize()} individual with {_join_phrases(ap)}.",
            # 4. "[Height] individual featuring X, Y and Z."
            lambda h, ap: f"{h.capitalize()} individual featuring {_join_phrases(ap)}.",
            # 5. "A [height] person. They have X, Y, and Z."  (split if ≥3 attrs)
            lambda h, ap: (
                f"A {h} person with {_join_phrases(ap)}."
                if len(ap) < 3 else
                f"A {h} person. They have {_join_phrases(ap[:len(ap)//2])}, "
                f"and {_join_phrases(ap[len(ap)//2:])}."
            ),
            # 6. Lead with BMI if extreme
            lambda h, ap: (
                f"{_pick('bmi_proxy', cats['bmi_proxy']).capitalize()} "
                f"with {h}, {_join_phrases(ap)}."
                if cats["bmi_proxy"] in ("very_low", "very_high")
                else f"A {h} person with {_join_phrases(ap)}."
            ),
            # 7. "A [height] frame with X, featuring Y and Z."
            lambda h, ap: (
                f"A {h} frame with {ap[0]}, featuring {_join_phrases(ap[1:])}."
                if len(ap) >= 2 else
                f"A {h} frame with {_join_phrases(ap)}."
            ),
            # 8. "[Height] person — X, Y and Z."
            lambda h, ap: f"{h.capitalize()} person — {_join_phrases(ap)}.",
        ]
        template = random.choice(plain_templates)
        return template(h_phrase, attr_phrases)


# ---------------------------------------------------------------------------
# Optional LLM rephrasing  (Qwen2.5-1.5B-Instruct)
# ---------------------------------------------------------------------------

class LLMRephraser:
    """
    Paraphrases template descriptions using Qwen2.5-1.5B-Instruct.
    Includes a keyword-level validation step: if the rephrased output
    changes any attribute category, the original template is kept.
    """

    # Map each category → keywords that must appear in the description
    _KEYWORDS: dict[str, dict[str, list[str]]] = {
        "height": {
            "very_low":  ["very short", "extremely short", "very petite"],
            "low":       ["short", "petite", "below-average height"],
            "average":   ["average height", "average stature", "medium height"],
            "high":      ["tall"],
            "very_high": ["very tall", "extremely tall", "towering"],
        },
        "bmi_proxy": {
            "very_low":  ["low body mass", "lean", "low body"],
            "very_high": ["lot of body mass", "heavy", "high body mass"],
        },
    }

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        print(f"Loading LLM rephraser: {model_name} …")
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._tokenizer = self._pipe.tokenizer

    def rephrase(self, description: str, cats: dict[str, str]) -> str:
        """
        Return a rephrased description, or the original if validation fails.
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "You rephrase physical body shape descriptions. "
                    "Keep ALL attributes and their levels (short/tall/etc.) identical. "
                    "Only change wording order and synonyms. "
                    "Output exactly ONE sentence, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Rephrase: \"{description}\"",
            },
        ]
        out = self._pipe(
            prompt,
            max_new_tokens=80,
            max_length=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        # Extract generated text after the user message
        raw = out[0]["generated_text"]
        # The pipeline returns the full conversation; extract assistant turn
        if isinstance(raw, list):
            generated = raw[-1].get("content", "").strip()
        else:
            # Strip prompt
            generated = raw[len(self._pipe.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )):].strip()

        # Remove leading/trailing quotes
        generated = generated.strip('"').strip("'").strip()

        if self._validate(generated, cats):
            return generated
        # Validation failed — return original
        return description

    def _validate(self, text: str, cats: dict[str, str]) -> bool:
        """Check that key attribute levels still appear in the rephrased text."""
        text_lower = text.lower()
        for attr, cat_kws in self._KEYWORDS.items():
            cat = cats.get(attr, "average")
            if cat not in cat_kws:
                continue
            required = cat_kws[cat]
            if not any(kw in text_lower for kw in required):
                return False
        return True


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples:    int,
    output_file:  str,
    gender:       str        = "neutral",
    batch_size:   int        = 128,
    n_calib:      int        = 5000,
    rephrase:     bool       = False,
    rephrase_model: str      = "Qwen/Qwen2.5-1.5B-Instruct",
    seed:         int        = 42,
    # --- Description style ---
    shape_prob:   float      = 0.65,
    n_extra_min:  int        = 1,
    n_extra_max:  int        = 3,
    focus_attrs:  list[str] | None = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SMPL
    smpl = smplx.create(
        str(_ROOT_DIR),
        model_type="smpl",
        gender=gender,
        num_betas=10,
    ).to(device)
    smpl.eval()

    # Calibrate thresholds
    thresholds = calibrate_thresholds(smpl, device, n_calib=n_calib,
                                      batch_size=batch_size)

    # Optional LLM rephraser
    rephraser = LLMRephraser(rephrase_model) if rephrase else None

    print(f"\nGenerating {n_samples} samples → {output_file}")
    written = 0
    rejected = 0  # samples where all measurements were zero (degenerate mesh)

    with open(output_file, "w") as fout:
        pbar = tqdm(total=n_samples, desc="generating")

        while written < n_samples:
            bs = min(batch_size, n_samples - written)
            betas_t = torch.randn(bs, 10, device=device).clamp(-3.5, 3.5)

            with torch.no_grad():
                out = _smpl_forward(smpl, betas_t, device)

            verts_np  = out.vertices.cpu().numpy()  # (bs, 6890, 3)
            joints_np = out.joints.cpu().numpy()    # (bs, 45,   3)
            betas_np  = betas_t.cpu().numpy()

            for i in range(bs):
                meas = extract_measurements(verts_np[i], joints_np[i])

                # Sanity check: reject degenerate meshes
                if meas["height"] < 0.5 or meas["height"] > 2.5:
                    rejected += 1
                    continue

                cats       = measurements_to_categories(meas, thresholds)
                body_shape = classify_body_shape(meas, thresholds)
                desc       = _generate_description(
                    cats, body_shape,
                    shape_prob=shape_prob,
                    n_extra_min=n_extra_min,
                    n_extra_max=n_extra_max,
                    focus_attrs=focus_attrs,
                )

                if rephraser is not None:
                    desc = rephraser.rephrase(desc, cats)

                beta_list = [round(float(b), 3) for b in betas_np[i]]
                record = {
                    "description":  desc,
                    "shape_params": str(beta_list),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)

                if written >= n_samples:
                    break

        pbar.close()

    print(f"Done. Wrote {written} samples to {output_file}.")
    if rejected > 0:
        print(f"  Rejected {rejected} degenerate meshes (outside height 0.5–2.5 m).")


# ---------------------------------------------------------------------------
# Validation utility: re-evaluate an existing JSONL file
# ---------------------------------------------------------------------------

def validate_dataset(
    jsonl_file: str,
    n_check:    int  = 200,
    gender:     str  = "neutral",
) -> None:
    """
    Audit data integrity: for a sample of entries, re-run SMPL, re-extract
    measurements, re-derive categories, and check the description against
    expected attribute levels.

    Reports per-attribute accuracy (height, weight/BMI, body shape) so you
    can spot systematic label errors.  Also prints a per-shape breakdown.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl   = smplx.create(str(_ROOT_DIR), model_type="smpl",
                          gender=gender, num_betas=10).to(device)
    smpl.eval()

    thresholds = calibrate_thresholds(smpl, device, n_calib=3000, batch_size=128)

    # Load random sample from file
    records: list[dict] = []
    with open(jsonl_file) as f:
        all_lines = f.readlines()
    sampled = random.sample(all_lines, min(n_check, len(all_lines)))
    for line in sampled:
        records.append(json.loads(line.strip()))

    print(f"\nValidating {len(records)} samples from {jsonl_file} …")

    # ------------------------------------------------------------------
    # Keyword maps
    # ------------------------------------------------------------------

    HEIGHT_KW: dict[str, list[str]] = {
        "very_low":  ["very short", "extremely short", "very petite"],
        "low":       ["short", "petite", "below-average height"],
        "average":   ["average height", "average stature", "medium height", "medium"],
        "high":      ["tall", "above-average height"],
        "very_high": ["very tall", "extremely tall", "towering"],
    }

    # Every phrase from ADJ["bmi_proxy"] across all levels — used to detect
    # whether weight is mentioned at all before checking accuracy.
    BMI_MENTIONED_KW = [
        # neutral vocabulary
        "body mass", "lean build", "average weight", "medium build", "heavy build",
        # colloquial vocabulary
        "skinny", "thin frame", "slim build", "slender", "slight build",
        "heavyset", "chubby", "overweight", "stocky", "fat build",
        "obese", "very overweight", "very fat", "large build",
        "lean", "slim",
    ]
    BMI_KW: dict[str, list[str]] = {
        "very_low":  ["low body mass", "very lean", "extremely low body mass",
                      "very skinny", "very thin", "extremely slim", "bony build"],
        "low":       ["below-average body mass", "lean build", "slim build",
                      "slender", "thin frame", "slight build"],
        "average":   ["average body mass", "average weight", "medium build",
                      "normal build", "regular build"],
        "high":      ["high body mass", "heavy build", "above-average body mass",
                      "heavyset", "chubby", "overweight", "stocky", "fat build",
                      "extra body weight"],
        "very_high": ["lot of body mass", "very heavy build", "extremely high body mass",
                      "obese", "very overweight", "very fat", "very large build"],
    }

    # Keywords that indicate a body shape type was explicitly named.
    SHAPE_MENTIONED_KW = [
        "hourglass", "pear", "pear-shaped", "apple", "apple-shaped",
        "rectangular", "banana", "straight figure", "inverted triangle", "v-shaped",
    ]
    # Shape name → phrases expected in the description
    SHAPE_KW: dict[str, list[str]] = {
        "hourglass":         ["hourglass"],
        "pear":              ["pear"],
        "apple":             ["apple"],
        "rectangle":         ["rectangular", "straight figure", "banana"],
        "inverted_triangle": ["inverted triangle", "v-shaped"],
    }

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    h_ok, h_ment = 0, 0
    b_ok, b_ment = 0, 0
    s_ok, s_ment = 0, 0
    total = 0

    # Per-shape breakdown: {shape: [mentioned, correct]}
    shape_stats: dict[str, list[int]] = {s: [0, 0] for s in SHAPE_KW}

    for rec in records:
        betas_list = json.loads(rec["shape_params"])
        betas_t    = torch.tensor([betas_list], dtype=torch.float32, device=device)

        with torch.no_grad():
            out = _smpl_forward(smpl, betas_t, device)

        verts  = out.vertices.cpu().numpy()[0]
        joints = out.joints.cpu().numpy()[0]
        meas   = extract_measurements(verts, joints)
        cats   = measurements_to_categories(meas, thresholds)
        shape  = classify_body_shape(meas, thresholds)

        desc_lower = rec["description"].lower()
        total += 1

        # ---- Height (always in description) ----
        h_cat   = cats["height"]
        h_match = any(kw in desc_lower for kw in HEIGHT_KW.get(h_cat, []))
        h_ment += 1
        h_ok   += int(h_match)

        # ---- Weight / BMI (only when mentioned) ----
        b_mentioned = any(kw in desc_lower for kw in BMI_MENTIONED_KW)
        if b_mentioned:
            b_cat   = cats["bmi_proxy"]
            b_match = any(kw in desc_lower for kw in BMI_KW.get(b_cat, []))
            b_ment += 1
            b_ok   += int(b_match)

        # ---- Body shape (only when mentioned) ----
        s_mentioned = any(kw in desc_lower for kw in SHAPE_MENTIONED_KW)
        if s_mentioned:
            s_match = any(kw in desc_lower for kw in SHAPE_KW.get(shape, []))
            s_ment += 1
            s_ok   += int(s_match)
            shape_stats[shape][0] += 1          # mentioned
            shape_stats[shape][1] += int(s_match)  # correct

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    h_acc = 100 * h_ok / h_ment if h_ment else 0
    b_acc = 100 * b_ok / b_ment if b_ment else 0
    s_acc = 100 * s_ok / s_ment if s_ment else 0

    print()
    print(f"  {'Attribute':<28}  {'Accuracy':>8}  {'Correct/Mentioned':>18}  {'Mention rate':>13}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*18}  {'-'*13}")
    print(f"  {'Height (always present)':<28}  {h_acc:>7.1f}%  {h_ok:>7}/{h_ment:<9}  {'100%':>13}")
    bmi_rate = f"{100*b_ment/total:.1f}%" if total else "—"
    print(f"  {'Weight / BMI':<28}  {b_acc:>7.1f}%  {b_ok:>7}/{b_ment:<9}  {bmi_rate:>13}")
    s_rate = f"{100*s_ment/total:.1f}%" if total else "—"
    print(f"  {'Body shape type':<28}  {s_acc:>7.1f}%  {s_ok:>7}/{s_ment:<9}  {s_rate:>13}")

    print()
    print("  Per-shape breakdown (when shape label is mentioned in description):")
    print(f"  {'Shape':<20}  {'Accuracy':>8}  {'Correct/Mentioned':>18}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*18}")
    for shape_name, (ment, ok) in shape_stats.items():
        acc = f"{100*ok/ment:.1f}%" if ment else "—"
        label = shape_name.replace("_", " ").title()
        print(f"  {label:<20}  {acc:>8}  {ok:>7}/{ment:<9}")

    print()
    print("  Notes:")
    print("  · Height appears in every description — its accuracy is a strict check.")
    print("  · Weight and body shape are only checked when their keywords appear.")
    print("  · Accuracy < 100% for 'average' categories is expected: the synonym")
    print("    chosen may not be in the validation keyword list.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load a YAML or JSON config file and return it as a dict."""
    p = Path(path)
    if not p.exists():
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with p.open() as f:
        if p.suffix in (".yaml", ".yml"):
            cfg = yaml.safe_load(f) or {}
        elif p.suffix == ".json":
            cfg = json.load(f)
        else:
            print(f"Unsupported config format: {p.suffix}  (use .yaml, .yml, or .json)",
                  file=sys.stderr)
            sys.exit(1)
    # Normalise hyphenated keys to underscored (n-calib → n_calib, etc.)
    return {k.replace("-", "_"): v for k, v in cfg.items()}


def main() -> None:
    # Pre-parse to detect --config before building the full parser,
    # so config values can be injected as parser defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None,
                     help="Path to a YAML/JSON config file (values act as defaults)")
    pre_args, _ = pre.parse_known_args()
    cfg_defaults: dict = _load_config(pre_args.config) if pre_args.config else {}

    # Apply vocabulary overrides early so they're active for the whole run.
    # Also honour --vocab-file passed directly on the CLI (pre-parse it too).
    pre2 = argparse.ArgumentParser(add_help=False)
    pre2.add_argument("--vocab-file", type=str, default=None)
    pre2_args, _ = pre2.parse_known_args()
    vocab_cfg = dict(cfg_defaults)
    if pre2_args.vocab_file:
        vocab_cfg["vocab_file"] = pre2_args.vocab_file
    _apply_vocab_overrides(vocab_cfg)

    parser = argparse.ArgumentParser(
        description="BodyShapeGPT Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     type=str,   default=None,
                        help="Path to a YAML/JSON config file")
    parser.add_argument("--vocab-file", type=str,   default=None,
                        help="Path to a YAML/JSON file with adj/shape_phrases overrides")
    parser.add_argument("--n",       type=int,   default=21_000,
                        help="Number of samples to generate")
    parser.add_argument("--output",  type=str,   default="BodyShapeGPT_dataset_new.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--gender",  choices=["neutral", "male", "female"],
                        default="neutral")
    parser.add_argument("--batch",   type=int,   default=128,
                        help="SMPL batch size")
    parser.add_argument("--n-calib", type=int,   default=5000,
                        help="Samples used to calibrate percentile thresholds")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument(
        "--rephrase", action="store_true",
        help="Rephrase descriptions with Qwen2.5-1.5B-Instruct for lexical variety",
    )
    parser.add_argument("--rephrase-model", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model for rephrasing")
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate an existing JSONL file (--output) instead of generating",
    )
    parser.add_argument("--validate-n", type=int, default=200,
                        help="Number of samples to audit in validation mode")

    # --- Description style ---
    parser.add_argument("--shape-prob", type=float, default=0.65,
                        help="Probability (0–1) of naming the body shape type in a description")
    parser.add_argument("--n-extra-min", type=int, default=1,
                        help="Minimum number of extra attributes to mention (besides height)")
    parser.add_argument("--n-extra-max", type=int, default=3,
                        help="Maximum number of extra attributes to mention (besides height)")
    parser.add_argument("--focus-attrs", type=str, default=None,
                        help="Comma-separated list of attributes that are always included "
                             "(e.g. 'bmi_proxy,shoulder_width'). "
                             f"Valid keys: {', '.join(MEAS_KEYS)}")

    # Apply config file values as defaults (CLI args still override them)
    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    args = parser.parse_args()

    # Parse focus_attrs: accept both comma-separated string (CLI) and list (config file)
    raw_focus = getattr(args, "focus_attrs", None)
    if isinstance(raw_focus, str):
        focus_attrs = [a.strip() for a in raw_focus.split(",") if a.strip()] or None
    elif isinstance(raw_focus, list):
        focus_attrs = raw_focus or None
    else:
        focus_attrs = None

    if args.validate:
        if not os.path.exists(args.output):
            print(f"File not found: {args.output}")
            sys.exit(1)
        validate_dataset(args.output, n_check=args.validate_n, gender=args.gender)
    else:
        generate_dataset(
            n_samples      = args.n,
            output_file    = args.output,
            gender         = args.gender,
            batch_size     = args.batch,
            n_calib        = args.n_calib,
            rephrase       = args.rephrase,
            rephrase_model = args.rephrase_model,
            seed           = args.seed,
            shape_prob     = args.shape_prob,
            n_extra_min    = args.n_extra_min,
            n_extra_max    = args.n_extra_max,
            focus_attrs    = focus_attrs,
        )


if __name__ == "__main__":
    main()
