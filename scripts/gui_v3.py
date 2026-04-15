"""
gui_v3.py — BodyShapeGPT Beta Slider GUI

Adjust each of the 10 SMPL shape parameters (betas) with drag sliders and
see the 3-D body mesh update in real time on the right.
"""

import sys
import os
import threading
import traceback
import argparse

import torch
import numpy as np
import smplx
import pyvista as pv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QSlider, QDoubleSpinBox, QGridLayout,
    QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt5.QtGui import QFont
from pyvistaqt import QtInteractor

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)

DEFAULT_GENDER = "neutral"

BETA_RANGE = 5.0          # sliders cover [-BETA_RANGE, +BETA_RANGE]
SLIDER_SCALE = 100        # int steps per unit → 0.01 precision
SLIDER_MIN = int(-BETA_RANGE * SLIDER_SCALE)
SLIDER_MAX = int( BETA_RANGE * SLIDER_SCALE)

RENDER_DEBOUNCE_MS = 80   # ms to wait after last slider move before rendering


# ---------------------------------------------------------------------------
# Qt worker signals
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    ready = pyqtSignal()
    error = pyqtSignal(str)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class BetaSliderGUI(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.smpl_model = None
        self._sliders: list[QSlider] = []
        self._spinboxes: list[QDoubleSpinBox] = []
        self._ready = False

        self.signals = WorkerSignals()
        self.signals.ready.connect(self._on_ready)
        self.signals.error.connect(self._on_error)

        # Debounce timer so we don't re-render on every pixel of drag
        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_mesh)

        self._init_ui()
        threading.Thread(target=self._bg_load, daemon=True).start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("BodyShapeGPT — Beta Slider")
        self.resize(1120, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---- Left panel ----
        left = QFrame()
        left.setFixedWidth(400)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        bold12 = QFont("Arial", 12, QFont.Bold)
        bold10 = QFont("Arial", 10, QFont.Bold)

        title = QLabel("BodyShapeGPT — Beta Slider")
        title.setFont(bold12)
        ll.addWidget(title)

        self.status_label = QLabel("Status: Loading SMPL model…")
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        ll.addWidget(self.status_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        ll.addWidget(sep)

        params_title = QLabel("Shape Parameters (β₀ – β₉)")
        params_title.setFont(bold10)
        ll.addWidget(params_title)

        # Scroll area for sliders (in case window is short)
        scroll_widget = QWidget()
        grid = QGridLayout(scroll_widget)
        grid.setContentsMargins(0, 0, 4, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)   # slider column stretches

        for i in range(10):
            # Beta label
            lbl = QLabel(f"β{i}")
            lbl.setFont(QFont("Courier", 10, QFont.Bold))
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setFixedWidth(24)

            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(SLIDER_MIN)
            slider.setMaximum(SLIDER_MAX)
            slider.setValue(0)
            slider.setTickInterval(SLIDER_SCALE)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            slider.setEnabled(False)

            # Spin box for numeric display / fine adjustment
            spin = QDoubleSpinBox()
            spin.setRange(-BETA_RANGE, BETA_RANGE)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(0.0)
            spin.setFixedWidth(72)
            spin.setEnabled(False)
            spin.setStyleSheet("QDoubleSpinBox { font-family: Courier; font-size: 10px; }")

            # Connections (block sibling to avoid feedback loops)
            slider.valueChanged.connect(
                lambda val, idx=i: self._slider_changed(idx, val)
            )
            spin.valueChanged.connect(
                lambda val, idx=i: self._spin_changed(idx, val)
            )

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

        # Range hint
        hint = QLabel(f"Drag range: [{-BETA_RANGE:.0f}, +{BETA_RANGE:.0f}]  |  step: 0.01")
        hint.setFont(QFont("Arial", 8))
        hint.setStyleSheet("color: #888;")
        ll.addWidget(hint)

        # Reset button
        self.reset_btn = QPushButton("Reset All to Zero")
        self.reset_btn.setMinimumHeight(36)
        self.reset_btn.setStyleSheet(
            "background-color: #555; color: white; font-weight: bold;"
        )
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._reset_all)
        ll.addWidget(self.reset_btn)

        # ---- Right panel (3-D viewer) ----
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
    # Slider / spinbox synchronisation
    # ------------------------------------------------------------------

    def _slider_changed(self, idx: int, int_val: int):
        float_val = int_val / SLIDER_SCALE
        spin = self._spinboxes[idx]
        spin.blockSignals(True)
        spin.setValue(float_val)
        spin.blockSignals(False)
        self._schedule_render()

    def _spin_changed(self, idx: int, float_val: float):
        slider = self._sliders[idx]
        int_val = int(round(float_val * SLIDER_SCALE))
        slider.blockSignals(True)
        slider.setValue(int_val)
        slider.blockSignals(False)
        self._schedule_render()

    def _schedule_render(self):
        """Restart the debounce timer; render fires once movement stops."""
        if self._ready:
            self._render_timer.start(RENDER_DEBOUNCE_MS)

    def _reset_all(self):
        for slider in self._sliders:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        for spin in self._spinboxes:
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._render_mesh()

    # ------------------------------------------------------------------
    # Background: load SMPL model
    # ------------------------------------------------------------------

    def _bg_load(self):
        try:
            self.smpl_model = smplx.create(
                model_path=_ROOT_DIR,
                model_type="smpl",
                gender=self.args.gender,
            )
            self.signals.ready.emit()
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def _on_ready(self):
        self.status_label.setText("Status: Ready — drag the sliders to reshape the body.")
        self.status_label.setStyleSheet("color: #2b78e4;")
        for s in self._sliders:
            s.setEnabled(True)
        for sp in self._spinboxes:
            sp.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self._ready = True
        self._render_mesh()   # show neutral (all-zero) body on startup

    def _on_error(self, msg: str):
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # SMPL run + render
    # ------------------------------------------------------------------

    def _get_betas(self) -> list:
        return [s.value() / SLIDER_SCALE for s in self._sliders]

    def _render_mesh(self):
        if self.smpl_model is None:
            return

        betas = self._get_betas()
        beta_t = torch.tensor([betas], dtype=torch.float32)

        with torch.no_grad():
            out = self.smpl_model(betas=beta_t, return_verts=True)

        vertices = out.vertices.detach().cpu().numpy().squeeze()
        faces    = self.smpl_model.faces

        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
        mesh = pv.PolyData(vertices, faces_pv)

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
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BodyShapeGPT — Interactive Beta Slider GUI"
    )
    parser.add_argument(
        "--gender", choices=["male", "female", "neutral"], default=DEFAULT_GENDER,
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = BetaSliderGUI(args)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
