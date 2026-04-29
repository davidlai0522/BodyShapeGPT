"""
gui_v4.py — BodyShapeGPT Diffusion Inference GUI

Same layout as gui_v3.py (sliders + 3-D viewer) but the betas come from the
text-conditioned diffusion model (train_v4.py) rather than manual slider input.

Left panel:
  • Text description input
  • n_samples + guidance-scale controls
  • Generate button
  • Sample navigator  (◀ Prev | Sample N/M | Next ▶)
  • Read-only beta sliders showing the current sample
  • Reset button (zero body)

Right panel:
  • Live PyVista 3-D mesh

Usage:
    python gui_v4.py [--gender neutral|male|female]
                     [--checkpoint ./weights/smpl_diffusion_v4]
                     [--cfg-scale 2.0]
                     [--ddim-steps 50]
"""

import sys
import os
import json
import threading
import traceback
import argparse
from datetime import datetime

import torch
import numpy as np
import smplx
import pyvista as pv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QSlider, QDoubleSpinBox, QGridLayout,
    QScrollArea, QSizePolicy, QTextEdit, QSpinBox, QGroupBox, QButtonGroup,
    QFileDialog, QMessageBox,
)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt5.QtGui import QFont
from pyvistaqt import QtInteractor

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)

DEFAULT_GENDER    = "neutral"
DEFAULT_CKPT_DIR  = os.path.join(_ROOT_DIR, "weights/smpl_diffusion_v4_1")
DEFAULT_CFG_SCALE = 2.0
DEFAULT_DDIM      = 50

BETA_RANGE    = 5.0
SLIDER_SCALE  = 100
SLIDER_MIN    = int(-BETA_RANGE * SLIDER_SCALE)
SLIDER_MAX    = int( BETA_RANGE * SLIDER_SCALE)
RENDER_DEBOUNCE_MS = 80


# ---------------------------------------------------------------------------
# Qt worker signals
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    models_ready   = pyqtSignal()                # SMPL + diffusion loaded
    smpl_ready     = pyqtSignal()                # SMPL reloaded (gender change)
    samples_ready  = pyqtSignal(object)          # list[np.ndarray] each (10,)
    error          = pyqtSignal(str)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DiffusionGUI(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.smpl_model  = None
        self._sample_fn  = None          # callable from train_v4.load_model_for_inference
        self._ready      = False

        self._samples: list[np.ndarray] = []   # list of (10,) arrays
        self._sample_idx: int = 0

        self._sliders:   list[QSlider]       = []
        self._spinboxes: list[QDoubleSpinBox] = []

        self.signals = WorkerSignals()
        self.signals.models_ready.connect(self._on_models_ready)
        self.signals.smpl_ready.connect(self._on_smpl_ready)
        self.signals.samples_ready.connect(self._on_samples_ready)
        self.signals.error.connect(self._on_error)

        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_mesh)

        self._init_ui()
        threading.Thread(target=self._bg_load, daemon=True).start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("BodyShapeGPT — Diffusion v4")
        self.resize(1200, 760)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ----------------------------------------------------------------
        # LEFT PANEL
        # ----------------------------------------------------------------
        left = QFrame()
        left.setFixedWidth(430)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        bold12 = QFont("Arial", 12, QFont.Bold)
        bold10 = QFont("Arial", 10, QFont.Bold)
        mono10 = QFont("Courier", 10, QFont.Bold)

        # Title
        title = QLabel("BodyShapeGPT — Diffusion v4")
        title.setFont(bold12)
        ll.addWidget(title)

        # Status
        self.status_label = QLabel("Status: Loading models…")
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        ll.addWidget(self.status_label)

        ll.addWidget(_hline())

        # ---- Gender toggle ----
        gender_row = QHBoxLayout()
        gender_row.setSpacing(6)
        gender_row.addWidget(QLabel("Gender:"))

        self._gender_btn_group = QButtonGroup(self)
        self._gender_btn_group.setExclusive(True)

        _GENDER_STYLE_ACTIVE  = (
            "background-color: #2b78e4; color: white; font-weight: bold;"
        )
        _GENDER_STYLE_INACTIVE = (
            "background-color: #ddd; color: #333;"
        )

        self._gender_btns: dict[str, QPushButton] = {}
        for label, key in [("Male", "male"), ("Female", "female"), ("Neutral", "neutral")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setChecked(key == self.args.gender)
            btn.setStyleSheet(
                _GENDER_STYLE_ACTIVE if key == self.args.gender else _GENDER_STYLE_INACTIVE
            )
            btn.toggled.connect(
                lambda checked, k=key, sa=_GENDER_STYLE_ACTIVE, si=_GENDER_STYLE_INACTIVE,
                       b=None: self._on_gender_toggled(k, checked, sa, si)
            )
            self._gender_btn_group.addButton(btn)
            gender_row.addWidget(btn)
            self._gender_btns[key] = btn

        gender_row.addStretch()
        ll.addLayout(gender_row)

        ll.addWidget(_hline())

        # ---- Generation group ----
        gen_box = QGroupBox("Text Description")
        gen_box.setFont(bold10)
        gen_layout = QVBoxLayout(gen_box)
        gen_layout.setSpacing(6)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "e.g. A tall person with broad shoulders and long arms."
        )
        self.text_input.setFixedHeight(80)
        self.text_input.setEnabled(False)
        gen_layout.addWidget(self.text_input)

        # Controls row: n_samples + guidance scale
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        ctrl_row.addWidget(QLabel("Samples:"))
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(1, 20)
        self.n_samples_spin.setValue(5)
        self.n_samples_spin.setFixedWidth(56)
        self.n_samples_spin.setEnabled(False)
        ctrl_row.addWidget(self.n_samples_spin)

        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(QLabel("Guidance:"))
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 8.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setDecimals(1)
        self.cfg_spin.setValue(self.args.cfg_scale)
        self.cfg_spin.setFixedWidth(60)
        self.cfg_spin.setEnabled(False)
        self.cfg_spin.setToolTip(
            "Classifier-free guidance scale.\n"
            "Higher → text description followed more strictly.\n"
            "Lower → more diverse / creative samples."
        )
        ctrl_row.addWidget(self.cfg_spin)
        ctrl_row.addStretch()
        gen_layout.addLayout(ctrl_row)

        # Generate button
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setMinimumHeight(36)
        self.gen_btn.setStyleSheet(
            "background-color: #2b78e4; color: white; font-weight: bold; font-size: 13px;"
        )
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self._on_generate)
        gen_layout.addWidget(self.gen_btn)

        ll.addWidget(gen_box)

        ll.addWidget(_hline())

        # ---- Sample navigator ----
        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setFixedWidth(80)
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._prev_sample)
        nav_row.addWidget(self.prev_btn)

        self.sample_counter = QLabel("—")
        self.sample_counter.setAlignment(Qt.AlignCenter)
        self.sample_counter.setFont(QFont("Arial", 10))
        nav_row.addWidget(self.sample_counter, stretch=1)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setFixedWidth(80)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._next_sample)
        nav_row.addWidget(self.next_btn)

        ll.addLayout(nav_row)

        ll.addWidget(_hline())

        # ---- Beta sliders (read-only display) ----
        params_title = QLabel("Shape Parameters (β₀ – β₉)")
        params_title.setFont(bold10)
        ll.addWidget(params_title)

        scroll_widget = QWidget()
        grid = QGridLayout(scroll_widget)
        grid.setContentsMargins(0, 0, 4, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)

        for i in range(10):
            lbl = QLabel(f"β{i}")
            lbl.setFont(mono10)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setFixedWidth(24)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(SLIDER_MIN)
            slider.setMaximum(SLIDER_MAX)
            slider.setValue(0)
            slider.setTickInterval(SLIDER_SCALE)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            slider.setEnabled(False)   # read-only — values set by model
            slider.valueChanged.connect(
                lambda val, idx=i: self._slider_changed(idx, val)
            )

            spin = QDoubleSpinBox()
            spin.setRange(-BETA_RANGE, BETA_RANGE)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(0.0)
            spin.setFixedWidth(72)
            spin.setEnabled(False)     # read-only
            spin.setStyleSheet("QDoubleSpinBox { font-family: Courier; font-size: 10px; }")

            grid.addWidget(lbl,    i, 0)
            grid.addWidget(slider, i, 1)
            grid.addWidget(spin,   i, 2)

            self._sliders.append(slider)
            self._spinboxes.append(spin)

        scroll = QScrollArea()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        ll.addWidget(scroll)

        hint = QLabel(f"Range: [{-BETA_RANGE:.0f}, +{BETA_RANGE:.0f}]  (read-only — set by model)")
        hint.setFont(QFont("Arial", 8))
        hint.setStyleSheet("color: #888;")
        ll.addWidget(hint)

        # Bottom button row: Reset + Save
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.reset_btn = QPushButton("Reset All to Zero")
        self.reset_btn.setMinimumHeight(36)
        self.reset_btn.setStyleSheet(
            "background-color: #555; color: white; font-weight: bold;"
        )
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._reset_all)
        btn_row.addWidget(self.reset_btn)

        self.save_btn = QPushButton("Save Betas…")
        self.save_btn.setMinimumHeight(36)
        self.save_btn.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold;"
        )
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Save current shape parameters to a JSON file")
        self.save_btn.clicked.connect(self._save_betas)
        btn_row.addWidget(self.save_btn)

        ll.addLayout(btn_row)

        # ----------------------------------------------------------------
        # RIGHT PANEL — 3-D viewer
        # ----------------------------------------------------------------
        right = QFrame()
        right.setStyleSheet("background-color: #1a1a1a;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(right)
        self.plotter.set_background("#1a1a1a")
        rl.addWidget(self.plotter.interactor)

        main_layout.addWidget(left)
        main_layout.addWidget(right, stretch=1)

    # ------------------------------------------------------------------
    # Slider (read-only sync)
    # ------------------------------------------------------------------

    def _slider_changed(self, idx: int, int_val: int):
        """Keep spinbox in sync when slider is set programmatically."""
        spin = self._spinboxes[idx]
        spin.blockSignals(True)
        spin.setValue(int_val / SLIDER_SCALE)
        spin.blockSignals(False)

    def _set_betas(self, betas: np.ndarray):
        """Push a (10,) beta array into the slider/spinbox display."""
        for i, val in enumerate(betas):
            val = float(np.clip(val, -BETA_RANGE, BETA_RANGE))
            int_val = int(round(val * SLIDER_SCALE))

            self._sliders[i].blockSignals(True)
            self._sliders[i].setValue(int_val)
            self._sliders[i].blockSignals(False)

            self._spinboxes[i].blockSignals(True)
            self._spinboxes[i].setValue(val)
            self._spinboxes[i].blockSignals(False)

    def _reset_all(self):
        self._samples = []
        self._sample_idx = 0
        self._update_nav_buttons()
        zeros = np.zeros(10)
        self._set_betas(zeros)
        self._render_mesh_betas(zeros)
        self.save_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Save betas
    # ------------------------------------------------------------------

    def _save_betas(self):
        betas = self._get_betas_from_sliders()
        description = self.text_input.toPlainText().strip()

        default_name = "shape_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Shape Parameters",
            os.path.join(os.path.expanduser("~"), default_name),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return  # user cancelled

        payload = {
            "description": description,
            "gender":      self.args.gender,
            "betas":       [round(float(b), 4) for b in betas],
            "saved_at":    datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            self.status_label.setText(f"Saved → {os.path.basename(path)}")
            self.status_label.setStyleSheet("color: #2e7d32;")
        except OSError as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    # ------------------------------------------------------------------
    # Sample navigation
    # ------------------------------------------------------------------

    def _update_nav_buttons(self):
        n = len(self._samples)
        if n == 0:
            self.sample_counter.setText("—")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.sample_counter.setText(f"Sample {self._sample_idx + 1} / {n}")
            self.prev_btn.setEnabled(self._sample_idx > 0)
            self.next_btn.setEnabled(self._sample_idx < n - 1)

    def _show_current_sample(self):
        if not self._samples:
            return
        betas = self._samples[self._sample_idx]
        self._set_betas(betas)
        self._render_mesh_betas(betas)
        self._update_nav_buttons()

    def _prev_sample(self):
        if self._sample_idx > 0:
            self._sample_idx -= 1
            self._show_current_sample()

    def _next_sample(self):
        if self._sample_idx < len(self._samples) - 1:
            self._sample_idx += 1
            self._show_current_sample()

    # ------------------------------------------------------------------
    # Gender toggle
    # ------------------------------------------------------------------

    _GENDER_STYLE_ACTIVE   = "background-color: #2b78e4; color: white; font-weight: bold;"
    _GENDER_STYLE_INACTIVE = "background-color: #ddd; color: #333;"

    def _on_gender_toggled(self, gender: str, checked: bool, sa: str, si: str):
        if not checked:
            self._gender_btns[gender].setStyleSheet(self._GENDER_STYLE_INACTIVE)
            return
        self._gender_btns[gender].setStyleSheet(self._GENDER_STYLE_ACTIVE)
        if self.args.gender == gender:
            return
        self.args.gender = gender
        self.status_label.setText(f"Status: Reloading SMPL ({gender})…")
        self.status_label.setStyleSheet("color: #666;")
        threading.Thread(target=self._bg_reload_smpl, daemon=True).start()

    def _bg_reload_smpl(self):
        try:
            self.smpl_model = smplx.create(
                model_path=_ROOT_DIR,
                model_type="smpl",
                gender=self.args.gender,
            )
            self.signals.smpl_ready.emit()
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def _on_smpl_ready(self):
        self.status_label.setText(
            f"Status: SMPL ({self.args.gender}) loaded."
        )
        self.status_label.setStyleSheet("color: #2b78e4;")
        # Re-render with current betas
        betas = self._get_betas_from_sliders()
        self._render_mesh_betas(betas)

    # ------------------------------------------------------------------
    # Background: load SMPL + diffusion model
    # ------------------------------------------------------------------

    def _bg_load(self):
        try:
            # 1. SMPL
            self.smpl_model = smplx.create(
                model_path=_ROOT_DIR,
                model_type="smpl",
                gender=self.args.gender,
            )

            # 2. Diffusion model (import here so GUI loads without it if absent)
            sys.path.insert(0, _SCRIPT_DIR)
            from train_v4 import load_model_for_inference
            _, _, _, self._sample_fn = load_model_for_inference(
                checkpoint_dir=self.args.checkpoint,
                cfg_scale=self.args.cfg_scale,
                ddim_steps=self.args.ddim_steps,
            )

            self.signals.models_ready.emit()
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def _on_models_ready(self):
        self.status_label.setText(
            "Status: Ready — describe a body shape and click Generate."
        )
        self.status_label.setStyleSheet("color: #2b78e4;")
        self.text_input.setEnabled(True)
        self.n_samples_spin.setEnabled(True)
        self.cfg_spin.setEnabled(True)
        self.gen_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self._ready = True
        self._render_mesh_betas(np.zeros(10))

    def _on_error(self, msg: str):
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # Diffusion inference (background thread)
    # ------------------------------------------------------------------

    def _on_generate(self):
        desc = self.text_input.toPlainText().strip()
        if not desc:
            self.status_label.setText("Please enter a description first.")
            self.status_label.setStyleSheet("color: #e47c2b;")
            return

        n  = self.n_samples_spin.value()
        w  = self.cfg_spin.value()

        self.gen_btn.setEnabled(False)
        self.status_label.setText(f"Generating {n} sample(s)…")
        self.status_label.setStyleSheet("color: #666;")

        threading.Thread(
            target=self._bg_generate,
            args=(desc, n, w),
            daemon=True,
        ).start()

    def _bg_generate(self, description: str, n_samples: int, cfg_scale: float):
        try:
            # Temporarily override cfg_scale if different from loaded model
            # (load_model_for_inference bakes it in; we re-call with new scale)
            sys.path.insert(0, _SCRIPT_DIR)
            from train_v4 import load_model_for_inference
            _, _, _, sample_fn = load_model_for_inference(
                checkpoint_dir=self.args.checkpoint,
                cfg_scale=cfg_scale,
                ddim_steps=self.args.ddim_steps,
            )
            raw = sample_fn(description, n_samples=n_samples)  # (n_samples, 10)
            samples = [raw[i] for i in range(n_samples)]
            self.signals.samples_ready.emit(samples)
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def _on_samples_ready(self, samples: list):
        self._samples    = samples
        self._sample_idx = 0
        self.gen_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_label.setText(
            f"Generated {len(samples)} sample(s). Use ◀ ▶ to browse."
        )
        self.status_label.setStyleSheet("color: #2b78e4;")
        self._show_current_sample()

    # ------------------------------------------------------------------
    # SMPL render
    # ------------------------------------------------------------------

    def _get_betas_from_sliders(self) -> np.ndarray:
        return np.array([s.value() / SLIDER_SCALE for s in self._sliders])

    def _render_mesh(self):
        """Debounced — used only when sliders change."""
        self._render_mesh_betas(self._get_betas_from_sliders())

    def _render_mesh_betas(self, betas: np.ndarray):
        if self.smpl_model is None:
            return
        beta_t = torch.tensor([betas], dtype=torch.float32)
        with torch.no_grad():
            out = self.smpl_model(betas=beta_t, return_verts=True)

        vertices = out.vertices.detach().cpu().numpy().squeeze()
        faces    = self.smpl_model.faces
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
        mesh     = pv.PolyData(vertices, faces_pv)

        self.plotter.clear_actors()
        self.plotter.add_mesh(
            mesh,
            color="#ecdbcd",
            pbr=True,
            metallic=0.1,
            roughness=0.5,
            smooth_shading=True,
        )
        if hasattr(self.plotter, "enable_ssao"):
            self.plotter.enable_ssao(radius=0.5, bias=0.01)
        self.plotter.reset_camera()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BodyShapeGPT — Diffusion v4 Inference GUI"
    )
    parser.add_argument(
        "--gender", choices=["male", "female", "neutral"], default=DEFAULT_GENDER,
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CKPT_DIR,
        help="Path to the weights/smpl_diffusion_v4 checkpoint directory",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=DEFAULT_CFG_SCALE,
        help="Default classifier-free guidance scale (overridable in GUI)",
    )
    parser.add_argument(
        "--ddim-steps", type=int, default=DEFAULT_DDIM,
        help="Number of DDIM denoising steps",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = DiffusionGUI(args)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
